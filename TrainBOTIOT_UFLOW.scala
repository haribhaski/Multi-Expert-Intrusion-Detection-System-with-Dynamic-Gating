import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object TrainBotIOT_UFlow_RF {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TrainBotIOT_UFlow_RF")
      .getOrCreate()

    import spark.implicits._

    val inPath  = "hdfs://namenode:8020/datasets/processed/uflow/botiot/uflow_100k_v1"
    val outPath = "hdfs://namenode:8020/models/uflow/botiot/rf_uflow_v1"

    // ========= Load =========
    val df0 = spark.read.parquet(inPath)
      .select(
        col("duration_s").cast("double"),
        col("src_bytes").cast("double"),
        col("dst_bytes").cast("double"),
        col("total_bytes").cast("double"),
        col("bytes_ratio").cast("double"),
        col("bytes_per_sec").cast("double"),
        col("log_duration").cast("double"),
        col("log_src_bytes").cast("double"),
        col("proto_tcp").cast("double"),
        col("proto_udp").cast("double"),
        col("proto_icmp").cast("double"),
        col("label").cast("double")
      )
      .na.fill(0.0)
      .persist(StorageLevel.MEMORY_AND_DISK)

    println(s"[BOTIOT-UFLOW] rows=${df0.count()} cols=${df0.columns.mkString(",")}")
    df0.groupBy("label").count().orderBy("label").show(false)

    // ========= Split =========
    val Array(train0, val0, test0) = df0.randomSplit(Array(0.7, 0.15, 0.15), seed = 42)
    val train = train0.persist(StorageLevel.MEMORY_AND_DISK)
    val valid = val0.persist(StorageLevel.MEMORY_AND_DISK)
    val test  = test0.persist(StorageLevel.MEMORY_AND_DISK)

    println(s"[SPLIT] train=${train.count()} val=${valid.count()} test=${test.count()}")

    // ========= Features =========
    val featureCols = Array(
      "duration_s","src_bytes","dst_bytes",
      "total_bytes","bytes_ratio","bytes_per_sec",
      "log_duration","log_src_bytes",
      "proto_tcp","proto_udp","proto_icmp"
    )

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // ========= Model =========
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(300)
      .setMaxDepth(12)
      .setMaxBins(128)
      .setFeatureSubsetStrategy("sqrt")
      .setSubsamplingRate(0.8)
      .setSeed(42)

    val pipeline = new Pipeline().setStages(Array(assembler, rf))

    // ========= Train =========
    val model = pipeline.fit(train)
    model.write.overwrite().save(outPath)
    println("[SAVED MODEL] " + outPath)

    // ========= Eval helper =========
    def eval(name: String, ds: org.apache.spark.sql.DataFrame): Unit = {
      val pred = model.transform(ds)
        .select(col("prediction").cast("double"), col("label").cast("double"))
        .na.drop()
        .persist(StorageLevel.MEMORY_AND_DISK)

      val rdd = pred.select("prediction","label").as[(Double, Double)].rdd.map { case (p,l) => (p,l) }
      val metrics = new MulticlassMetrics(rdd)

      val cm = metrics.confusionMatrix
      val acc = metrics.accuracy

      val p0 = metrics.precision(0.0); val r0 = metrics.recall(0.0); val f0 = metrics.fMeasure(0.0)
      val p1 = metrics.precision(1.0); val r1 = metrics.recall(1.0); val f1 = metrics.fMeasure(1.0)

      val macroP = (p0 + p1) / 2.0
      val macroR = (r0 + r1) / 2.0
      val macroF = (f0 + f1) / 2.0

      println(s"\n================ $name ================")
      println("Confusion Matrix (rows=true label, cols=pred):")
      println(cm)
      println(f"Accuracy     : $acc%.6f")
      println(f"Precision(0) : $p0%.6f   Recall(0): $r0%.6f   F1(0): $f0%.6f")
      println(f"Precision(1) : $p1%.6f   Recall(1): $r1%.6f   F1(1): $f1%.6f")
      println(f"Macro-Prec   : $macroP%.6f")
      println(f"Macro-Recall : $macroR%.6f")
      println(f"Macro-F1     : $macroF%.6f")

      pred.unpersist()
    }

    eval("VALIDATION", valid)
    eval("TEST", test)

    // ========= Keep Spark UI alive =========
    // 10 minutes = 600000 ms
    Thread.sleep(600000)

    // cleanup
    train.unpersist(); valid.unpersist(); test.unpersist(); df0.unpersist()
    spark.stop()
  }
}