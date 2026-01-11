import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object ImprovedEnsemble_RF_LR_GBT_SAFE {

  case class Weights(wRf: Double, wLr: Double, wGbt: Double) {
    def normalized: Weights = {
      val sum = wRf + wLr + wGbt
      Weights(wRf / sum, wLr / sum, wGbt / sum)
    }
  }

  case class SamplingConfig(
    normal: Double = 1.0,
    dos: Double = 1.0,
    probe: Double = 1.0,
    r2l: Double = 0.25,
    u2r: Double = 0.15
  )

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("NSL-KDD Ensemble SAFE: RF(MC)+LR(MC)+GBT(Binary) -> Binary")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.default.parallelism", "8")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .config("spark.kryoserializer.buffer.max", "512m")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val dataBase = if (args.length > 0) args(0) else "hdfs://namenode:8020/datasets/processed/nsl-kdd"
    val outBase  = if (args.length > 1) args(1) else "hdfs://namenode:8020/models/nsl-kdd_ensemble_safe"
    val autoTune = if (args.length > 2) args(2).toBoolean else false

    spark.sparkContext.setCheckpointDir(s"$outBase/_checkpoints")

    val CACHE = StorageLevel.MEMORY_AND_DISK

    println("=" * 90)
    println("SAFE ENSEMBLE: RF(Multi) + LR(Multi) + GBT(Binary) -> Binary Predictions")
    println("=" * 90)

    // ============================================================
    // [1] Load minimal columns (NO cache yet)
    // ============================================================
    println("\n[1/10] Load data...")
    val requiredCols = Seq("label", "attack_category", "features")

    val trainRaw0 = spark.read.parquet(s"$dataBase/train_parquet")
      .select(requiredCols.map(col): _*)
      .filter(col("attack_category") =!= "unknown")
      .withColumn("label", col("label").cast("double"))

    val testRaw = spark.read.parquet(s"$dataBase/test_parquet")
      .select(requiredCols.map(col): _*)
      .withColumn("label", col("label").cast("double"))

    // Cache only for distribution + sampling stage
    val trainRaw = trainRaw0.persist(CACHE)
    val trainCount = trainRaw.count()
    println(s"Loaded $trainCount training samples")

    // ============================================================
    // [2] Distribution Analysis (cached DF)
    // ============================================================
    println("\n[2/10] Class distribution...")
    val origCountsDF = trainRaw.groupBy("attack_category").count().cache()
    val origCounts = origCountsDF.collect().map(r => r.getString(0) -> r.getLong(1)).toMap
    val origTotal = origCounts.values.sum.toDouble
    val numClasses = origCounts.size.toDouble

    println(s"Total: ${origTotal.toLong}")
    origCounts.toSeq.sortBy(-_._2).foreach { case (c, n) =>
      println(f"  $c%-12s: $n%,10d (${n / origTotal * 100}%.2f%%)")
    }

    // Also compute true binary label counts from original data (more correct)
    val binCountsRaw = trainRaw.groupBy("label").count().collect().map(r => r.getDouble(0) -> r.getLong(1)).toMap
    val n0 = binCountsRaw.getOrElse(0.0, 1L).toDouble
    val n1 = binCountsRaw.getOrElse(1.0, 1L).toDouble
    val totalBin = n0 + n1
    val w0 = totalBin / (2.0 * n0)
    val w1 = totalBin / (2.0 * n1)
    println(f"\nBinary label counts: normal=$n0%.0f attack=$n1%.0f")
    println(f"Binary weights: w0=$w0%.4f w1=$w1%.4f")

    // ============================================================
    // [3] Stratified resampling (bounded) + SINGLE checkpoint
    // ============================================================
    println("\n[3/10] Stratified resampling (bounded, memory-safe)...")
    val maxCount = origCounts.values.max.toDouble
    val config = SamplingConfig()

    val target = Map(
      "normal" -> maxCount * config.normal,
      "dos"    -> maxCount * config.dos,
      "probe"  -> maxCount * config.probe,
      "r2l"    -> maxCount * config.r2l,
      "u2r"    -> maxCount * config.u2r
    )

    val sampledParts: Seq[DataFrame] = origCounts.keys.toSeq.map { cat =>
      val cnt = origCounts(cat).toDouble
      val tgt = target.getOrElse(cat, cnt)
      val frac = math.min(if (cnt > 0) tgt / cnt else 1.0, 6.0)

      println(f"  $cat%-12s: ${cnt.toLong}%,10d -> ${tgt.toLong}%,10d (frac=$frac%.2f)")

      val df = trainRaw.filter(col("attack_category") === cat)
      if (frac >= 1.0) df.sample(withReplacement = true, frac, seed = 42)
      else df.sample(withReplacement = false, frac, seed = 42)
    }

    val stratifiedTrain = sampledParts.reduce(_ union _)
      .repartition(8)
      .persist(CACHE)

    val stratN = stratifiedTrain.count()
    println(s"Resampled total: $stratN")

    // Checkpoint to cut lineage once
    val stratifiedCp = stratifiedTrain.checkpoint(eager = true)
    stratifiedTrain.unpersist()

    // Free early caches
    trainRaw.unpersist()
    origCountsDF.unpersist()

    // ============================================================
    // [4] Index labels (fit on stratified)
    // ============================================================
    println("\n[4/10] Indexing attack categories...")
    val indexer = new StringIndexer()
      .setInputCol("attack_category")
      .setOutputCol("category_label")
      .setHandleInvalid("keep")

    val indexerModel = indexer.fit(stratifiedCp)
    val labels = indexerModel.labelsArray(0)
    println(s"Classes: ${labels.length}")
    labels.zipWithIndex.foreach { case (lab, i) => println(f"  $i%2d -> $lab") }

    val trainIndexed = indexerModel.transform(stratifiedCp)
      .select(col("features"), col("label"), col("category_label"))
      .persist(CACHE)

    trainIndexed.count()
    stratifiedCp.unpersist()

    // ============================================================
    // [5] Weights (MC from original distribution) + build trainMC/trainBIN
    // ============================================================
    println("\n[5/10] Computing weights + building train sets...")

    val categoryWeights = origCounts.map { case (cat, cnt) =>
      cat -> (origTotal / (numClasses * cnt.toDouble))
    }

    val labelWeightMap = labels.zipWithIndex.map { case (cat, idx) =>
      idx.toDouble -> categoryWeights.getOrElse(cat, 1.0)
    }.toMap

    val bMcW = spark.sparkContext.broadcast(labelWeightMap)
    val addMcWeight = udf((lbl: Double) => bMcW.value.getOrElse(lbl, 1.0))

    val addBinWeight = udf((y: Double) => if (y == 1.0) w1 else w0)

    val trainMC = trainIndexed
      .withColumn("mcWeight", addMcWeight(col("category_label")))
      .select(col("features"), col("category_label"), col("mcWeight"))
      .persist(CACHE)

    val trainBIN = trainIndexed
      .withColumn("binWeight", addBinWeight(col("label")))
      .select(col("features"), col("label"), col("binWeight"))
      .persist(CACHE)

    trainMC.count(); trainBIN.count()
    trainIndexed.unpersist()

    // ============================================================
    // [6] Train models (SAFE params) + safe autoTune option
    // ============================================================
    println("\n[6/10] Training models (SAFE settings)...")

    val rfBase = new RandomForestClassifier()
      .setLabelCol("category_label")
      .setFeaturesCol("features")
      .setWeightCol("mcWeight")
      .setSeed(42)
      // SAFE knobs (same spirit as your low-mem RF that succeeded)
      .setNumTrees(80)
      .setMaxDepth(12)
      .setMaxBins(16)
      .setMinInstancesPerNode(10)
      .setMinInfoGain(0.004)
      .setSubsamplingRate(0.6)
      .setFeatureSubsetStrategy("sqrt")
      .setCacheNodeIds(false)
      .setCheckpointInterval(10)

    val lrBase = new LogisticRegression()
      .setLabelCol("category_label")
      .setFeaturesCol("features")
      .setWeightCol("mcWeight")
      .setMaxIter(200)
      .setRegParam(0.02)
      .setElasticNetParam(0.0)
      .setStandardization(true)
      .setFamily("multinomial")
      .setTol(1e-6)

    val gbtBase = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("binWeight")
      .setSeed(42)
      // SAFE knobs
      .setMaxIter(80)
      .setMaxDepth(6)
      .setMaxBins(16)
      .setStepSize(0.08)
      .setSubsamplingRate(0.7)
      .setFeatureSubsetStrategy("sqrt")
      .setCheckpointInterval(10)
      .setMaxMemoryInMB(512)

    val t0 = System.currentTimeMillis()

    val rfModel: RandomForestClassificationModel =
      if (!autoTune) rfBase.fit(trainMC)
      else {
        println("autoTune=true -> SAFE mini CV on sampled tune set (2 folds, tiny grid)...")

        val tune = trainMC.sample(withReplacement = false, 0.30, seed = 42).cache()
        tune.count()

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("category_label")
          .setMetricName("f1")

        val grid = new ParamGridBuilder()
          .addGrid(rfBase.numTrees, Array(60, 80))
          .addGrid(rfBase.maxDepth, Array(10, 12))
          .build()

        val cv = new CrossValidator()
          .setEstimator(rfBase)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(grid)
          .setNumFolds(2)
          .setSeed(42)

        val best = cv.fit(tune).bestModel.asInstanceOf[RandomForestClassificationModel]
        tune.unpersist()
        best
      }

    val lrModel = lrBase.fit(trainMC)
    val gbtModel = gbtBase.fit(trainBIN)

    val trainTime = (System.currentTimeMillis() - t0) / 1000.0
    println(f"✓ Trained models in $trainTime%.1f seconds")

    trainMC.unpersist()
    trainBIN.unpersist()

    // ============================================================
    // [7] Predict on test set (single rid) + compute ensemble
    // ============================================================
    println("\n[7/10] Predict + ensemble on test...")

    val testIndexed = indexerModel.transform(testRaw)
      .select(col("label"), col("attack_category"), col("category_label"), col("features"))
      .withColumn("rid", monotonically_increasing_id())
      .persist(CACHE)
    testIndexed.count()

    val pVecToArr = udf((v: Vector) => v.toArray.toSeq)
    val p1 = udf((v: Vector) => v(1))

    val prf = rfModel.transform(testIndexed)
      .select(col("rid"), pVecToArr(col("probability")).alias("p_rf_mc"))

    val plr = lrModel.transform(testIndexed)
      .select(col("rid"), pVecToArr(col("probability")).alias("p_lr_mc"))

    val pgbt = gbtModel.transform(testIndexed)
      .select(col("rid"), p1(col("probability")).alias("p_gbt_attack"))

    val normalIdx = labels.indexOf("normal")
    val bNormalIdx = spark.sparkContext.broadcast(normalIdx)

    val mcAttackScore = udf((arr: Seq[Double]) => {
      val idx = bNormalIdx.value
      if (idx >= 0 && arr != null && arr.length > idx) 1.0 - arr(idx) else 0.5
    })

    // Weights (keep your “original feature” of weight struct)
    val W = Weights(wRf = 0.40, wLr = 0.25, wGbt = 0.35).normalized
    println(f"Ensemble weights: RF=${W.wRf}%.3f, LR=${W.wLr}%.3f, GBT=${W.wGbt}%.3f")

    val predictions = testIndexed
      .select(col("rid"), col("label"), col("attack_category"), col("category_label"))
      .join(prf, "rid")
      .join(plr, "rid")
      .join(pgbt, "rid")
      .withColumn("score_rf", mcAttackScore(col("p_rf_mc")))
      .withColumn("score_lr", mcAttackScore(col("p_lr_mc")))
      .withColumn("binary_score",
        lit(W.wRf) * col("score_rf") +
        lit(W.wLr) * col("score_lr") +
        lit(W.wGbt) * col("p_gbt_attack")
      )
      .withColumn("binary_pred", when(col("binary_score") >= 0.5, 1.0).otherwise(0.0))
      .withColumn("binary_label", when(col("attack_category") === "normal", 0.0).otherwise(1.0))
      .persist(CACHE)

    predictions.count()
    testIndexed.unpersist()

    // ============================================================
    // [8] Evaluate binary
    // ============================================================
    println("\n[8/10] Evaluating...")
    val rdd = predictions.select("binary_pred", "binary_label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)

    println("\n" + "=" * 90)
    println("ENSEMBLE BINARY RESULTS")
    println("=" * 90)
    println(f"Accuracy:  ${metrics.accuracy * 100}%.2f%%")
    println(f"Precision: ${metrics.precision(1.0) * 100}%.2f%%")
    println(f"Recall:    ${metrics.recall(1.0) * 100}%.2f%%")
    println(f"F1:        ${metrics.fMeasure(1.0) * 100}%.2f%%")
    println(f"WeightedF1:${metrics.weightedFMeasure * 100}%.2f%%")
    println(s"\nConfusion Matrix:\n${metrics.confusionMatrix}")

    // ============================================================
    // [9] Save
    // ============================================================
    println("\n[9/10] Saving...")
    rfModel.write.overwrite().save(s"$outBase/rf_multiclass")
    lrModel.write.overwrite().save(s"$outBase/lr_multiclass")
    gbtModel.write.overwrite().save(s"$outBase/gbt_binary")
    indexerModel.write.overwrite().save(s"$outBase/category_indexer")

    Seq((W.wRf, W.wLr, W.wGbt)).toDF("w_rf", "w_lr", "w_gbt")
      .write.mode("overwrite").parquet(s"$outBase/ensemble_weights")

    predictions.select(
      col("label").alias("orig_label"),
      col("attack_category"),
      col("binary_label"),
      col("binary_pred"),
      col("binary_score"),
      col("score_rf"),
      col("score_lr"),
      col("p_gbt_attack")
    ).write.mode("overwrite").parquet(s"$outBase/test_predictions")

    // Save metrics
    Seq((
      metrics.accuracy,
      metrics.precision(1.0),
      metrics.recall(1.0),
      metrics.fMeasure(1.0),
      metrics.weightedFMeasure,
      trainTime
    )).toDF("accuracy", "precision", "recall", "f1", "weighted_f1", "train_time_sec")
      .write.mode("overwrite").parquet(s"$outBase/metrics")

    // ============================================================
    // [10] Cleanup
    // ============================================================
    println("\n[10/10] Cleanup...")
    predictions.unpersist()

    println("DONE ✅")
    spark.stop()
  }
}
