import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, PolynomialExpansion}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel

object TrainLR_Poly_NoLeak_Safe {

  case class ThRes(th: Double, f1: Double, precision: Double, recall: Double)

  // Driver-side threshold search (fast, no repeated Spark jobs)
  def bestThresholdF1Local(scoresAndLabels: Array[(Double, Double)]): ThRes = {
    val thresholds = (1 to 99).map(_ / 100.0)
    var bestTh = 0.5
    var bestF1 = -1.0
    var bestPrec = 0.0
    var bestRec = 0.0

    thresholds.foreach { th =>
      var tp = 0L; var fp = 0L; var fn = 0L
      var i = 0
      while (i < scoresAndLabels.length) {
        val s = scoresAndLabels(i)._1
        val y = scoresAndLabels(i)._2
        val p = if (s >= th) 1.0 else 0.0
        if (p == 1.0 && y == 1.0) tp += 1
        else if (p == 1.0 && y == 0.0) fp += 1
        else if (p == 0.0 && y == 1.0) fn += 1
        i += 1
      }
      val prec = if (tp + fp == 0) 0.0 else tp.toDouble / (tp + fp)
      val rec  = if (tp + fn == 0) 0.0 else tp.toDouble / (tp + fn)
      val f1   = if (prec + rec == 0) 0.0 else 2.0 * prec * rec / (prec + rec)
      if (f1 > bestF1) { 
        bestF1 = f1
        bestTh = th
        bestPrec = prec
        bestRec = rec
      }
    }
    ThRes(bestTh, bestF1, bestPrec, bestRec)
  }

  def evaluate(df: DataFrame, predCol: String, scoreCol: String, name: String): Unit = {
    println(s"\n$name:")
    println("-" * 70)

    val predAndLabels = df.select(col(predCol).cast("double"), col("label").cast("double"))
      .rdd.map(r => (r.getDouble(0), r.getDouble(1)))

    val mm = new MulticlassMetrics(predAndLabels)
    println(f"Accuracy:  ${mm.accuracy * 100}%.2f%%")
    println(f"Precision: ${mm.precision(1.0) * 100}%.2f%%")
    println(f"Recall:    ${mm.recall(1.0) * 100}%.2f%%")
    println(f"F1-Score:  ${mm.fMeasure(1.0) * 100}%.2f%%")
    println(s"\nConfusion Matrix:\n${mm.confusionMatrix}")

    val scoreAndLabels = df.select(col(scoreCol).cast("double"), col("label").cast("double"))
      .rdd.map(r => (r.getDouble(0), r.getDouble(1)))

    val bm = new BinaryClassificationMetrics(scoreAndLabels)
    println(f"ROC-AUC: ${bm.areaUnderROC() * 100}%.2f%%")
    println(f"PR-AUC:  ${bm.areaUnderPR() * 100}%.2f%%")
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("NSL-KDD LR+Poly (NO LEAKAGE, SAFE)")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val dataBase = if (args.length > 0) args(0) else "hdfs://namenode:8020/datasets/processed/nsl-kdd"
    val outBase  = if (args.length > 1) args(1) else "hdfs://namenode:8020/models/nsl-kdd_lr_poly_safe"

    val trainPath = s"$dataBase/train_parquet"
    val testPath  = s"$dataBase/test_parquet"

    val CACHE = StorageLevel.DISK_ONLY
    val getP1 = udf((p: Vector) => p(1))

    println("=" * 90)
    println("NSL-KDD: LR + ChiSq + Poly2 (NO TEST LEAKAGE, MEMORY SAFE)")
    println("=" * 90)
    println(s"Train parquet: $trainPath")
    println(s"Test parquet : $testPath")
    println(s"Output base  : $outBase")
    println("=" * 90)

    println("\n[1/7] Load data...")
    val trainAll = spark.read.parquet(trainPath)
    val testData = spark.read.parquet(testPath)
    println(s"Train samples: ${trainAll.count()}")
    println(s"Test samples : ${testData.count()}")

    println("\n[2/7] Split TRAIN into Train/Val (85/15) - NO TEST TOUCHING...")
    val Array(trainSet, valSet) = trainAll.randomSplit(Array(0.85, 0.15), seed = 42)
    val trainP = trainSet.persist(CACHE)
    val valP   = valSet.persist(CACHE)
    println(s"TrainSet: ${trainP.count()} | ValSet: ${valP.count()}")

    println("\n[3/7] Compute class weight from TRAIN and add weightCol...")
    val pos = trainP.filter($"label" === 1.0).count().toDouble
    val neg = trainP.filter($"label" === 0.0).count().toDouble
    val posWeight = if (pos == 0) 1.0 else (neg / pos)
    println(f"Train class distribution: pos=$pos%.0f, neg=$neg%.0f")
    println(f"Computed posWeight = neg/pos = $posWeight%.3f")

    val trainW = trainP.withColumn("weight", 
      when($"label" === 1.0, lit(posWeight)).otherwise(lit(1.0))).persist(CACHE)
    trainW.count()
    trainP.unpersist()

    println("\n[4/7] Manual grid search on TRAIN→VAL (memory-safe)...")

    // SAFE grid for laptop:
    // - ChiSqSelector reduces to K features BEFORE poly
    // - Poly degree fixed to 2 (degree 3 explodes memory)
    // - Example: 64 features → poly2 → ~2,080 features (manageable)
    //            128 features → poly2 → ~8,256 features (risky)
    
    val grid = Seq(
      // (ChiSqK, RegParam)
      (48, 0.001),   // Medium features, light reg
      (64, 0.001),   // More features, light reg
      (80, 0.001),   // High features, light reg
      (64, 0.003),   // Medium features, moderate reg
      (48, 0.003),   // Fewer features, moderate reg
      (96, 0.001),   // Maximum features, light reg
      (64, 0.01),    // Medium features, strong reg
      (80, 0.003)    // High features, moderate reg
    )

    var bestValF1 = -1.0
    var bestTh = 0.5
    var bestModel: PipelineModel = null
    var bestParams: (Int, Double) = null

    grid.zipWithIndex.foreach { case ((k, reg), idx) =>
      println(s"\n${"=" * 80}")
      println(f"Config ${idx + 1}/${grid.size}: ChiSqK=$k, RegParam=$reg")
      println("=" * 80)

      // Step 1: ChiSq feature selection (dimensionality reduction)
      val selector = new ChiSqSelector()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setOutputCol("sel_features")
        .setNumTopFeatures(k)

      // Step 2: Polynomial expansion degree 2 on reduced features
      val poly = new PolynomialExpansion()
        .setInputCol("sel_features")
        .setOutputCol("poly_features")
        .setDegree(2)  // ONLY degree 2 - safe for memory

      // Step 3: Logistic Regression with class weights
      val lr = new LogisticRegression()
        .setLabelCol("label")
        .setFeaturesCol("poly_features")
        .setWeightCol("weight")
        .setMaxIter(150)
        .setRegParam(reg)
        .setElasticNetParam(0.0)  // Pure L2
        .setTol(1e-6)

      val pipe = new Pipeline().setStages(Array(selector, poly, lr))
      
      try {
        println("Training...")
        val startTime = System.currentTimeMillis()
        val model = pipe.fit(trainW)
        val trainTime = (System.currentTimeMillis() - startTime) / 1000.0
        println(f"Training completed in $trainTime%.1f seconds")

        // Evaluate on VALIDATION set (from TRAIN split)
        println("Scoring validation set...")
        val valScores = model.transform(valP)
          .select(col("label").cast("double"), col("probability"))
          .withColumn("score", getP1(col("probability")))
          .select("score", "label")
          .rdd.map(r => (r.getDouble(0), r.getDouble(1)))
          .collect()

        val th = bestThresholdF1Local(valScores)
        println(f"VAL: th=${th.th}%.3f, F1=${th.f1 * 100}%.2f%%, Precision=${th.precision * 100}%.2f%%, Recall=${th.recall * 100}%.2f%%")

        if (th.f1 > bestValF1) {
          bestValF1 = th.f1
          bestTh = th.th
          if (bestModel != null) bestModel = null
          bestModel = model
          bestParams = (k, reg)
          println("*** NEW BEST MODEL ***")
        }
        
        System.gc()
        Thread.sleep(100)
        
      } catch {
        case e: Exception =>
          println(s"ERROR: ${e.getMessage}")
          e.printStackTrace()
      }
    }

    if (bestModel == null) {
      println("ERROR: No model succeeded!")
      spark.stop()
      return
    }

    println("\n" + "=" * 90)
    println(f"BEST MODEL: ChiSqK=${bestParams._1}, RegParam=${bestParams._2}, posWeight=$posWeight%.3f")
    println(f"BEST VAL: th=$bestTh%.3f, valF1=${bestValF1 * 100}%.2f%%")
    println("=" * 90)

    bestModel.write.overwrite().save(s"$outBase/best_lr_poly_pipeline")

    Seq(("lr_poly", bestTh, bestValF1, posWeight, bestParams._1, bestParams._2))
      .toDF("model", "best_threshold", "val_f1", "pos_weight", "chisq_k", "reg_param")
      .write.mode("overwrite")
      .parquet(s"$outBase/thresholds")

    trainW.unpersist()
    valP.unpersist()

    println("\n[5/7] TEST evaluation (clean, no leakage)...")
    val testScored = bestModel.transform(testData)
      .select("label", "attack_category", "probability")
      .withColumn("lr_score", getP1(col("probability")))
      .withColumn("lr_pred", when(col("lr_score") >= lit(bestTh), 1.0).otherwise(0.0))
      .persist(CACHE)

    testScored.count()

    println("\n" + "=" * 90)
    println("FINAL TEST SET EVALUATION (no leakage)")
    println("=" * 90)
    evaluate(testScored, "lr_pred", "lr_score", "LR+ChiSq+Poly2 (TEST)")

    println("\n[6/7] Save test predictions...")
    testScored.write.mode("overwrite").parquet(s"$outBase/test_predictions")

    println("\n[7/7] Cleanup...")
    testScored.unpersist()

    spark.stop()
    
    println("\n" + "=" * 90)
    println("DONE ✅")
    println("=" * 90)
    println("\nNo test leakage - proper train/val/test split")
    println("Memory-safe: ChiSq reduces dims before Poly2")
    println("Class weights computed from training data only")
    println("=" * 90)
  }
}