import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object ImprovedEnsemble_RF_LR_GBT_ULTRASAFE {

  case class Weights(wRf: Double, wLr: Double, wGbt: Double) {
    def normalized: Weights = {
      val s = wRf + wLr + wGbt
      Weights(wRf / s, wLr / s, wGbt / s)
    }
  }

  case class SamplingConfig(
    normal: Double = 1.0,
    dos: Double = 1.0,
    probe: Double = 1.0,
    r2l: Double = 0.25,
    u2r: Double = 0.15,
    maxFrac: Double = 4.0     // LOWER than 6.0 to reduce blow-up
  )

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("NSL-KDD Ensemble ULTRASAFE: RF(MC)+LR(MC)+GBT(BIN) -> BIN")
      .config("spark.sql.shuffle.partitions", "4")
      .config("spark.default.parallelism", "4")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val dataBase = if (args.length > 0) args(0) else "hdfs://namenode:8020/datasets/processed/nsl-kdd"
    val outBase  = if (args.length > 1) args(1) else "hdfs://namenode:8020/models/nsl-kdd_ensemble_ultrasafe"
    val autoTune = if (args.length > 2) args(2).toBoolean else false // kept, but we won’t CV here

    val CACHE = StorageLevel.MEMORY_AND_DISK_SER

    println("=" * 90)
    println("ULTRASAFE ENSEMBLE: RF(Multi) + LR(Multi) + GBT(Binary) -> Binary Predictions")
    println("=" * 90)

    // ============================================================
    // [1] Load
    // ============================================================
    println("\n[1/9] Load data...")
    val requiredCols = Seq("label", "attack_category", "features")

    val trainRaw = spark.read.parquet(s"$dataBase/train_parquet")
      .select(requiredCols.map(col): _*)
      .filter(col("attack_category") =!= "unknown")
      .withColumn("label", col("label").cast("double"))
      .persist(CACHE)

    val testRaw = spark.read.parquet(s"$dataBase/test_parquet")
      .select(requiredCols.map(col): _*)
      .withColumn("label", col("label").cast("double"))

    val trainCount = trainRaw.count()
    println(s"Train rows: $trainCount")

    // ============================================================
    // [2] Distribution (small collect)
    // ============================================================
    println("\n[2/9] Distribution...")
    val origCounts = trainRaw.groupBy("attack_category").count().collect()
      .map(r => r.getString(0) -> r.getLong(1)).toMap

    val origTotal = origCounts.values.sum.toDouble
    val numClasses = origCounts.size.toDouble

    val maxCount = origCounts.values.max.toDouble
    println(s"Total: ${origTotal.toLong}, maxClass: ${maxCount.toLong}")

    // true binary counts from label
    val binCounts = trainRaw.groupBy("label").count().collect()
      .map(r => r.getDouble(0) -> r.getLong(1).toDouble).toMap

    val n0 = binCounts.getOrElse(0.0, 1.0)
    val n1 = binCounts.getOrElse(1.0, 1.0)
    val totalBin = n0 + n1
    val w0 = totalBin / (2.0 * n0)
    val w1 = totalBin / (2.0 * n1)

    println(f"Binary counts: n0=$n0%.0f n1=$n1%.0f | w0=$w0%.4f w1=$w1%.4f")

    // ============================================================
    // [3] Stratified sampling (bounded LOWER) + repartition
    // ============================================================
    println("\n[3/9] Stratified sampling (bounded)...")
    val cfg = SamplingConfig()

    val target = Map(
      "normal" -> maxCount * cfg.normal,
      "dos"    -> maxCount * cfg.dos,
      "probe"  -> maxCount * cfg.probe,
      "r2l"    -> maxCount * cfg.r2l,
      "u2r"    -> maxCount * cfg.u2r
    )

    val sampledParts: Seq[DataFrame] = origCounts.keys.toSeq.map { cat =>
      val cnt = origCounts(cat).toDouble
      val tgt = target.getOrElse(cat, cnt)
      val frac = math.min(if (cnt > 0) tgt / cnt else 1.0, cfg.maxFrac)

      println(f"  $cat%-10s: ${cnt.toLong}%,8d -> ${tgt.toLong}%,8d (frac=$frac%.2f)")

      val df = trainRaw.filter(col("attack_category") === cat)
      if (frac >= 1.0) df.sample(withReplacement = true, frac, seed = 42)
      else df.sample(withReplacement = false, frac, seed = 42)
    }

    val stratified = sampledParts.reduce(_ union _)
      .repartition(4)
      .persist(CACHE)

    val stratN = stratified.count()
    println(s"Resampled rows: $stratN")

    // ============================================================
    // [4] Index multi-class labels (fit on stratified)
    // ============================================================
    println("\n[4/9] Indexing categories...")
    val indexerModel = new StringIndexer()
      .setInputCol("attack_category")
      .setOutputCol("category_label")
      .setHandleInvalid("keep")
      .fit(stratified)

    val labels = indexerModel.labelsArray(0)
    println(s"Classes=${labels.length}")
    labels.zipWithIndex.foreach { case (lab, i) => println(s"  $i -> $lab") }

    val normalIdx = labels.indexOf("normal")
    if (normalIdx < 0) println("WARN: 'normal' not found in labels. Binary conversion may be off.")

    // ============================================================
    // [5] Build trainMC, trainBIN sequentially (NO simultaneous caching)
    // ============================================================
    println("\n[5/9] Building trainMC (MC weights)...")

    val categoryWeights = origCounts.map { case (cat, cnt) =>
      cat -> (origTotal / (numClasses * cnt.toDouble))
    }

    val labelWeightMap = labels.zipWithIndex.map { case (cat, idx) =>
      idx.toDouble -> categoryWeights.getOrElse(cat, 1.0)
    }.toMap

    val bMcW = spark.sparkContext.broadcast(labelWeightMap)
    val addMcWeight = udf((lbl: Double) => bMcW.value.getOrElse(lbl, 1.0))
    val addBinWeight = udf((y: Double) => if (y == 1.0) w1 else w0)

    val indexed = indexerModel.transform(stratified)
      .select(col("features"), col("label"), col("category_label"))
      .persist(CACHE)
    indexed.count()

    // free stratified early
    stratified.unpersist()
    trainRaw.unpersist()

    val trainMC = indexed
      .withColumn("mcWeight", addMcWeight(col("category_label")))
      .select(col("features"), col("category_label"), col("mcWeight"))
      .persist(CACHE)
    trainMC.count()

    // ============================================================
    // [6] Train RF+LR on trainMC (ULTRASAFE params)
    // ============================================================
    println("\n[6/9] Training RF + LR (multi-class, ultrasafe)...")

    val rf = new RandomForestClassifier()
      .setLabelCol("category_label")
      .setFeaturesCol("features")
      .setWeightCol("mcWeight")
      .setSeed(42)
      .setNumTrees(60)
      .setMaxDepth(10)
      .setMaxBins(16)
      .setMinInstancesPerNode(12)
      .setMinInfoGain(0.005)
      .setSubsamplingRate(0.55)
      .setFeatureSubsetStrategy("sqrt")
      .setCacheNodeIds(false)
      .setCheckpointInterval(10)

    val lr = new LogisticRegression()
      .setLabelCol("category_label")
      .setFeaturesCol("features")
      .setWeightCol("mcWeight")
      .setMaxIter(150)
      .setRegParam(0.03)
      .setElasticNetParam(0.0)
      .setStandardization(true)
      .setFamily("multinomial")
      .setTol(1e-6)

    val rfModel = rf.fit(trainMC)
    val lrModel = lr.fit(trainMC)

    trainMC.unpersist()

    // ============================================================
    // [7] Train GBT on binary (build trainBIN now)
    // ============================================================
    println("\n[7/9] Training GBT (binary, ultrasafe)...")

    val trainBIN = indexed
      .withColumn("binWeight", addBinWeight(col("label")))
      .select(col("features"), col("label"), col("binWeight"))
      .persist(CACHE)
    trainBIN.count()

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("binWeight")
      .setSeed(42)
      .setMaxIter(60)
      .setMaxDepth(5)
      .setMaxBins(16)
      .setStepSize(0.08)
      .setSubsamplingRate(0.65)
      .setFeatureSubsetStrategy("sqrt")
      .setCheckpointInterval(10)
      .setMaxMemoryInMB(384)

    val gbtModel = gbt.fit(trainBIN)

    trainBIN.unpersist()
    indexed.unpersist()

    // ============================================================
    // [8] Predict on test + ensemble -> binary (NO persist unless needed)
    // ============================================================
    println("\n[8/9] Predict + ensemble -> binary...")

    val pVecToArr = udf((v: Vector) => v.toArray.toSeq)
    val p1 = udf((v: Vector) => v(1))

    val testIndexed = indexerModel.transform(testRaw)
      .select(col("label"), col("attack_category"), col("category_label"), col("features"))
      .withColumn("rid", monotonically_increasing_id())
      .repartition(4)
      .persist(CACHE)
    testIndexed.count()

    val prf = rfModel.transform(testIndexed)
      .select(col("rid"), pVecToArr(col("probability")).alias("p_rf_mc"))

    val plr = lrModel.transform(testIndexed)
      .select(col("rid"), pVecToArr(col("probability")).alias("p_lr_mc"))

    val pgbt = gbtModel.transform(testIndexed)
      .select(col("rid"), p1(col("probability")).alias("p_gbt_attack"))

    val bNormalIdx = spark.sparkContext.broadcast(normalIdx)
    val mcAttackScore = udf((arr: Seq[Double]) => {
      val idx = bNormalIdx.value
      if (idx >= 0 && arr != null && arr.length > idx) 1.0 - arr(idx) else 0.5
    })

    val W = Weights(0.40, 0.25, 0.35).normalized
    println(f"Weights: RF=${W.wRf}%.3f LR=${W.wLr}%.3f GBT=${W.wGbt}%.3f")

    val predictions = testIndexed
      .select(col("rid"), col("label"), col("attack_category"))
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

    // Metrics
    val rdd = predictions.select("binary_pred", "binary_label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    val m = new MulticlassMetrics(rdd)

    println("\n" + "=" * 90)
    println("ENSEMBLE BINARY RESULTS")
    println("=" * 90)
    println(f"Accuracy:  ${m.accuracy * 100}%.2f%%")
    println(f"Precision: ${m.precision(1.0) * 100}%.2f%%")
    println(f"Recall:    ${m.recall(1.0) * 100}%.2f%%")
    println(f"F1:        ${m.fMeasure(1.0) * 100}%.2f%%")
    println(s"\nConfusion Matrix:\n${m.confusionMatrix}")

    // ============================================================
    // [9] Save (write triggers action; keep test cached until write done)
    // ============================================================
    println("\n[9/9] Saving artifacts...")
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

    testIndexed.unpersist()

    println("DONE ✅")
    spark.stop()
  }
}
