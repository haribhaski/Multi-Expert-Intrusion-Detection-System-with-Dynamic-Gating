import org.apache.spark.sql.{SparkSession, DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object TrainFinalEnsembleStacking_NoLeak_WeightedGating {

  case class MetricsRow(
    run_id: String,
    holdout: String,
    split: String,
    src_ds: String,
    rows: Long,
    accuracy: Double,
    precision0: Double, recall0: Double, f10: Double,
    precision1: Double, recall1: Double, f11: Double,
    macroF1: Double,
    cm00: Double, cm01: Double, cm10: Double, cm11: Double
  )

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TrainFinalEnsembleStacking_NoLeak_WeightedGating")
      .getOrCreate()

    import spark.implicits._

    // ---------------- CONFIG ----------------
    val kFolds = if (args.length >= 1) args(0).toInt else 5
    val seed   = if (args.length >= 2) args(1).toLong else 42L

    // ---------------- INPUT U-FLOW DATASETS ----------------
    val nslPath = "hdfs://namenode:8020/datasets/processed/uflow/nslkdd/uflow_20k_v1"
    val cicPath = "hdfs://namenode:8020/datasets/processed/uflow/cicids2017/uflow_100k_v1"
    val botPath = "hdfs://namenode:8020/datasets/processed/uflow/botiot/uflow_100k_v1"

    // ---------------- EXPERT MODELS (PipelineModel only) ----------------
    val modelPathNSL = "hdfs://namenode:8020/models/experts/nslkdd/rf_uflow_v1"
    val modelPathCIC = "hdfs://namenode:8020/models/uflow/cicids2017/rf_uflow_v1"
    val modelPathBOT = "hdfs://namenode:8020/models/uflow/botiot/rf_uflow_v1"

    // ---------------- OUTPUTS (WRITE ONLY TO /user/spark/...) ----------------
    val outRootEnsemble = "hdfs://namenode:8020/user/spark/models/uflow/ensemble/stacking_lr_v3_weighted_gating"
    val outRootMetrics  = "hdfs://namenode:8020/user/spark/metrics/uflow/ensemble/stacking_lr_v3_weighted_gating"

    // ---------------- U-FLOW FEATURES ----------------
    val uflowCols = Array(
      "duration_s","src_bytes","dst_bytes",
      "total_bytes","bytes_ratio","bytes_per_sec",
      "log_duration","log_src_bytes",
      "proto_tcp","proto_udp","proto_icmp"
    )

    def loadUF(path: String, tag: String): DataFrame = {
      spark.read.parquet(path)
        .select((uflowCols.map(c => col(c).cast("double")) :+ col("label").cast("double")): _*)
        .na.fill(0.0)
        .withColumn("src_ds", lit(tag))
    }

    val nsl = loadUF(nslPath, "nsl")
    val cic = loadUF(cicPath, "cic")
    val bot = loadUF(botPath, "bot")

    val allRaw = nsl.unionByName(cic).unionByName(bot)
      .withColumn("row_id", monotonically_increasing_id())
      .persist(StorageLevel.MEMORY_AND_DISK)

    println(s"[ALL] rows=${allRaw.count()}")
    allRaw.groupBy("src_ds","label").count().orderBy("src_ds","label").show(false)

    // ---------------- LOAD EXPERTS ----------------
    val mNSL = PipelineModel.load(modelPathNSL)
    val mCIC = PipelineModel.load(modelPathCIC)
    val mBOT = PipelineModel.load(modelPathBOT)

    // Score ONE expert -> p(class=1)
    def withProb(df: DataFrame, model: PipelineModel, outCol: String): DataFrame = {
      val cleaned = df.drop("features", "rawPrediction", "probability", "prediction")
      val scored = model.transform(cleaned)
      scored
        .withColumn(outCol, vector_to_array(col("probability"))(1))
        .drop("rawPrediction", "probability", "prediction")
    }

    /**
     * TRUE GATING FEATURES:
     * keep uflowCols + probabilities
     */
    def scoreExpertsKeepUF(df: DataFrame): DataFrame = {
      val s1 = withProb(df, mNSL, "p_nsl")
      val s2 = withProb(s1, mCIC, "p_cic")
      val s3 = withProb(s2, mBOT, "p_bot")

      // keep row_id, src_ds, label, 3 probs + original U-Flow features
      val keepCols =
        Seq($"row_id", $"src_ds", $"label", $"p_nsl", $"p_cic", $"p_bot") ++ uflowCols.map(c => col(c))
      s3.select(keepCols: _*).na.fill(0.0)
    }

    // ---------------- METRICS ----------------
    def evalBinary(runId: String, holdout: String, splitName: String, dsName: String, dfPred: DataFrame): MetricsRow = {
      val pred = dfPred
        .select(col("prediction").cast("double").as("prediction"),
                col("label").cast("double").as("label"))
        .na.drop()

      val n = pred.count()
      if (n == 0L) {
        return MetricsRow(runId, holdout, splitName, dsName, 0L,
          0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0, 0,0,0,0)
      }

      val rdd = pred.as[(Double, Double)].rdd.map { case (p, l) => (p, l) }
      val m = new MulticlassMetrics(rdd)

      val acc = m.accuracy
      val p0 = m.precision(0.0); val r0 = m.recall(0.0); val f0 = m.fMeasure(0.0)
      val p1 = m.precision(1.0); val r1 = m.recall(1.0); val f1 = m.fMeasure(1.0)
      val macroF1 = (f0 + f1) / 2.0

      val cm = m.confusionMatrix
      val cm00 = cm(0,0); val cm01 = cm(0,1); val cm10 = cm(1,0); val cm11 = cm(1,1)

      MetricsRow(runId, holdout, splitName, dsName, n, acc, p0,r0,f0, p1,r1,f1, macroF1, cm00,cm01,cm10,cm11)
    }

    // ---------------- TRAIN/EVAL ----------------
    val holdouts = Seq("nsl","cic","bot")
    val runId = java.time.LocalDateTime.now().toString.replace(":","-")

    var metricsAcc = Vector.empty[MetricsRow]

    holdouts.foreach { holdoutDs =>
      println(s"\n==================== HOLDOUT = $holdoutDs ====================")

      val metaTrainRaw = allRaw.filter($"src_ds" =!= lit(holdoutDs)).persist(StorageLevel.MEMORY_AND_DISK)
      val metaTestRaw  = allRaw.filter($"src_ds" === lit(holdoutDs)).persist(StorageLevel.MEMORY_AND_DISK)

      println(s"[META TRAIN] rows=${metaTrainRaw.count()}  [META TEST($holdoutDs)] rows=${metaTestRaw.count()}")

      // deterministic folds
      val metaTrainF = metaTrainRaw
        .withColumn("fold", pmod(abs(xxhash64($"row_id")), lit(kFolds)))
        .persist(StorageLevel.MEMORY_AND_DISK)

      // -------- OOF stacking (no leakage) --------
      var oofAll: DataFrame = null

      (0 until kFolds).foreach { f =>
        val foldVal = metaTrainF.filter($"fold" === lit(f))
        val oofVal = scoreExpertsKeepUF(foldVal)
          .withColumn("split", lit("oof_train"))

        oofAll = if (oofAll == null) oofVal else oofAll.unionByName(oofVal)
      }

      oofAll = oofAll.persist(StorageLevel.MEMORY_AND_DISK)
      println(s"[OOF BUILT] rows=${oofAll.count()} folds=$kFolds")

      // -------- Class weights for meta LR (attack = label 1) --------
      val pos = oofAll.filter($"label" === 1.0).count().toDouble
      val neg = oofAll.filter($"label" === 0.0).count().toDouble
      val posWeight = if (pos <= 0.0) 1.0 else (neg / pos)

      val oofW = oofAll.withColumn(
        "weight",
        when($"label" === 1.0, lit(posWeight)).otherwise(lit(1.0))
      ).persist(StorageLevel.MEMORY_AND_DISK)

      println(f"[WEIGHTS] neg=$neg%.0f pos=$pos%.0f posWeight=$posWeight%.4f")

      // -------- META MODEL: probs + U-Flow features --------
      val metaInputs = Array("p_nsl","p_cic","p_bot") ++ uflowCols

      val metaAssembler = new VectorAssembler()
        .setInputCols(metaInputs)
        .setOutputCol("meta_features")

      val lr = new LogisticRegression()
        .setFeaturesCol("meta_features")
        .setLabelCol("label")
        .setWeightCol("weight")          // ✅ key fix
        .setMaxIter(200)

      val metaPipe = new Pipeline().setStages(Array(metaAssembler, lr))
      val metaModel = metaPipe.fit(oofW)

      val outEnsemble = s"$outRootEnsemble/holdout=$holdoutDs"
      metaModel.write.overwrite().save(outEnsemble)
      println(s"[SAVED META MODEL] $outEnsemble")

      // -------- Evaluate OOF train --------
      val oofPred = metaModel.transform(oofW).persist(StorageLevel.MEMORY_AND_DISK)
      metricsAcc = metricsAcc :+ evalBinary(runId, holdoutDs, "oof_train", "ALL", oofPred)

      val oofDsList = oofW.select("src_ds").distinct().as[String].collect().toSeq
      oofDsList.foreach { ds =>
        metricsAcc = metricsAcc :+ evalBinary(runId, holdoutDs, "oof_train", ds, oofPred.filter($"src_ds" === lit(ds)))
      }

      // -------- Evaluate TRUE holdout test --------
      val holdoutStacked = scoreExpertsKeepUF(metaTestRaw).withColumn("split", lit("holdout_test"))
        .persist(StorageLevel.MEMORY_AND_DISK)

      // add same weight col for consistency (not used in prediction)
      val holdoutW = holdoutStacked.withColumn("weight", lit(1.0)).persist(StorageLevel.MEMORY_AND_DISK)

      val holdoutPred = metaModel.transform(holdoutW).persist(StorageLevel.MEMORY_AND_DISK)
      metricsAcc = metricsAcc :+ evalBinary(runId, holdoutDs, "holdout_test", holdoutDs, holdoutPred)

      val key = metricsAcc.filter(m => m.holdout == holdoutDs && m.split == "holdout_test" && m.src_ds == holdoutDs)
      key.foreach { m =>
        println(s"\n[HOLDOUT RESULT] holdout=${m.holdout} rows=${m.rows}")
        println(f"Accuracy=${m.accuracy}%.6f  Macro-F1=${m.macroF1}%.6f")
        println(f"F1(0)=${m.f10}%.6f  F1(1)=${m.f11}%.6f")
        println(s"ConfusionMatrix=[[${m.cm00}, ${m.cm01}], [${m.cm10}, ${m.cm11}]]")
      }

      // cleanup
      oofPred.unpersist()
      holdoutPred.unpersist()
      holdoutW.unpersist()
      holdoutStacked.unpersist()
      oofW.unpersist()
      oofAll.unpersist()
      metaTrainF.unpersist()
      metaTrainRaw.unpersist()
      metaTestRaw.unpersist()
    }

    // ---------------- SAVE METRICS (no toDS) ----------------
    val schema = StructType(Seq(
      StructField("run_id", StringType, false),
      StructField("holdout", StringType, false),
      StructField("split", StringType, false),
      StructField("src_ds", StringType, false),
      StructField("rows", LongType, false),
      StructField("accuracy", DoubleType, false),
      StructField("precision0", DoubleType, false),
      StructField("recall0", DoubleType, false),
      StructField("f10", DoubleType, false),
      StructField("precision1", DoubleType, false),
      StructField("recall1", DoubleType, false),
      StructField("f11", DoubleType, false),
      StructField("macroF1", DoubleType, false),
      StructField("cm00", DoubleType, false),
      StructField("cm01", DoubleType, false),
      StructField("cm10", DoubleType, false),
      StructField("cm11", DoubleType, false)
    ))

    val rowsRdd = spark.sparkContext.parallelize(metricsAcc).map { m =>
      Row(m.run_id, m.holdout, m.split, m.src_ds, m.rows,
        m.accuracy,
        m.precision0, m.recall0, m.f10,
        m.precision1, m.recall1, m.f11,
        m.macroF1,
        m.cm00, m.cm01, m.cm10, m.cm11
      )
    }

    val metricsDF = spark.createDataFrame(rowsRdd, schema)
      .withColumn("saved_at", current_timestamp())

    println("\n==================== ALL METRICS ====================")
    metricsDF.orderBy(col("holdout"), col("split"), col("src_ds")).show(200, truncate = false)

    val outMetrics = s"$outRootMetrics/run_id=$runId"
    metricsDF.write.mode("overwrite").parquet(outMetrics)
    println(s"[SAVED METRICS] $outMetrics")

    allRaw.unpersist()
    spark.stop()
  }
}