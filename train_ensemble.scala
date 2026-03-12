import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object TrainFinalEnsembleStacking {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TrainFinalEnsembleStacking")
      .getOrCreate()

    import spark.implicits._

    // ================== INPUT U-FLOW DATASETS ==================
    val nslPath = "hdfs://namenode:8020/datasets/processed/uflow/nslkdd/uflow_20k_v1"
    val cicPath = "hdfs://namenode:8020/datasets/processed/uflow/cicids2017/uflow_100k_v1"
    val botPath = "hdfs://namenode:8020/datasets/processed/uflow/botiot/uflow_100k_v1"

    // ================== EXPERT MODELS (PIPELINE MODELS) ==================
    val modelPathNSL = "hdfs://namenode:8020/models/experts/nslkdd/rf_uflow_v1"
    val modelPathCIC = "hdfs://namenode:8020/models/uflow/cicids2017/rf_uflow_v1"
    val modelPathBOT = "hdfs://namenode:8020/models/uflow/botiot/rf_uflow_v1"

    // ================== OUTPUT (META MODEL) ==================
    val outEnsemble = "hdfs://namenode:8020/models/uflow/ensemble/stacking_lr_v1"

    // ================== U-FLOW FEATURE COLS ==================
    val featureCols = Array(
      "duration_s","src_bytes","dst_bytes",
      "total_bytes","bytes_ratio","bytes_per_sec",
      "log_duration","log_src_bytes",
      "proto_tcp","proto_udp","proto_icmp"
    )

    def loadUF(p: String): DataFrame = {
      spark.read.parquet(p)
        .select((featureCols.map(c => col(c).cast("double")) :+ col("label").cast("double")): _*)
        .na.fill(0.0)
    }

    // ================== BUILD TRAIN POOL ==================
    val nsl = loadUF(nslPath).withColumn("src_ds", lit("nsl"))
    val cic = loadUF(cicPath).withColumn("src_ds", lit("cic"))
    val bot = loadUF(botPath).withColumn("src_ds", lit("bot"))

    val all = nsl.unionByName(cic).unionByName(bot)
      .persist(StorageLevel.MEMORY_AND_DISK)

    println(s"[ALL] rows=${all.count()}")
    all.groupBy("src_ds","label").count().orderBy("src_ds","label").show(false)

    // ================== LOAD EXPERTS ==================
    val mNSL = PipelineModel.load(modelPathNSL)
    val mCIC = PipelineModel.load(modelPathCIC)
    val mBOT = PipelineModel.load(modelPathBOT)

    /**
     * Best-fix withProb:
     * - Drops "features" to avoid: Output column features already exists (because expert pipeline has its own assembler)
     * - Drops any leftover prediction columns from previous expert
     * - Extracts prob[1] robustly using vector_to_array
     */
    def withProb(df: DataFrame, model: PipelineModel, outCol: String): DataFrame = {
      val cleaned =
        df.drop("features", "rawPrediction", "probability", "prediction")

      val scored = model.transform(cleaned)

      // probability is VectorUDT -> convert to array then index 1
      scored
        .withColumn(outCol, vector_to_array(col("probability"))(1))
        .drop("rawPrediction", "probability", "prediction")
    }

    // ================== STACK FEATURES ==================
    val stacked1 = withProb(all, mNSL, "p_nsl")
    val stacked2 = withProb(stacked1, mCIC, "p_cic")
    val stacked3 = withProb(stacked2, mBOT, "p_bot")
      .select(col("label"), col("p_nsl"), col("p_cic"), col("p_bot"))
      .na.fill(0.0)
      .persist(StorageLevel.MEMORY_AND_DISK)

    println("[STACKED SAMPLE]")
    stacked3.show(5, false)

    // ================== SPLIT ==================
    val Array(train0, val0, test0) = stacked3.randomSplit(Array(0.7, 0.15, 0.15), seed = 42)
    val train = train0.persist(StorageLevel.MEMORY_AND_DISK)
    val valid = val0.persist(StorageLevel.MEMORY_AND_DISK)
    val test  = test0.persist(StorageLevel.MEMORY_AND_DISK)

    println(s"[SPLIT] train=${train.count()} val=${valid.count()} test=${test.count()}")

    // ================== META MODEL ==================
    val metaAssembler = new VectorAssembler()
      .setInputCols(Array("p_nsl","p_cic","p_bot"))
      .setOutputCol("meta_features")

    val lr = new LogisticRegression()
      .setFeaturesCol("meta_features")
      .setLabelCol("label")
      .setMaxIter(200)

    val metaPipe = new Pipeline().setStages(Array(metaAssembler, lr))
    val metaModel = metaPipe.fit(train)

    metaModel.write.overwrite().save(outEnsemble)
    println("[SAVED ENSEMBLE] " + outEnsemble)

    // ================== EVAL ==================
    def eval(name: String, ds: DataFrame): Unit = {
      val pred = metaModel.transform(ds)
        .select(col("prediction").cast("double").as("prediction"),
                col("label").cast("double").as("label"))
        .na.drop()

      val rdd = pred.as[(Double, Double)].rdd.map { case (p, l) => (p, l) }
      val metrics = new MulticlassMetrics(rdd)

      val acc = metrics.accuracy
      val p0 = metrics.precision(0.0); val r0 = metrics.recall(0.0); val f0 = metrics.fMeasure(0.0)
      val p1 = metrics.precision(1.0); val r1 = metrics.recall(1.0); val f1 = metrics.fMeasure(1.0)

      println(s"\n================ $name ================")
      println("Confusion Matrix (rows=true, cols=pred):")
      println(metrics.confusionMatrix)
      println(f"Accuracy     : $acc%.6f")
      println(f"Precision(0) : $p0%.6f   Recall(0): $r0%.6f   F1(0): $f0%.6f")
      println(f"Precision(1) : $p1%.6f   Recall(1): $r1%.6f   F1(1): $f1%.6f")
      println(f"Macro-F1     : ${(f0 + f1) / 2.0}%.6f")
    }

    eval("VALIDATION", valid)
    eval("TEST", test)

    // ================== CLEANUP ==================
    train.unpersist()
    valid.unpersist()
    test.unpersist()
    stacked3.unpersist()
    all.unpersist()

    // Keep Spark UI open (10 min)
    Thread.sleep(600000)
    spark.stop()
  }
}