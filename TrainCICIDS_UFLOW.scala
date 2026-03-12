// TrainCICIDS2017_UFlow_RF.scala
// RandomForest on CICIDS2017 U-FLOW parquet + full metrics + keep Spark UI

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}

object TrainCICIDS2017_UFlow_RF {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TrainCICIDS2017_UFlow_RF")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val inPath  = if (args.length > 0) args(0)
      else "hdfs://namenode:8020/datasets/processed/uflow/cicids2017/uflow_100k_v1"

    val modelOut = if (args.length > 1) args(1)
      else "hdfs://namenode:8020/models/uflow/cicids2017/rf_uflow_v1"

    // U-FLOW numeric feature cols (no protocol string)
    val featureCols = Array(
      "duration_s","src_bytes","dst_bytes",
      "total_bytes","bytes_ratio","bytes_per_sec",
      "log_duration","log_src_bytes",
      "proto_tcp","proto_udp","proto_icmp"
    )

    // ===== READ =====
    val df0 = spark.read.parquet(inPath)

    // Keep only needed cols + clean NaNs
    val data = df0
      .select((featureCols.map(c => col(c).cast("double")) :+ col("label").cast("double")): _*)
      .na.fill(0.0)
      .cache()

    println(s"[INFO] inPath=$inPath")
    println(s"[INFO] rows=${data.count()} cols=${data.columns.mkString(",")}")
    data.groupBy("label").count().orderBy("label").show(false)

    // ===== SPLIT =====
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 42)

    // ===== VECTORIZE =====
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    println(s"[INFO] train=${train.count()} test=${test.count()}")

    // ===== MODEL =====
    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setNumTrees(200)
      .setMaxDepth(14)
      .setMinInstancesPerNode(2)
      .setFeatureSubsetStrategy("sqrt")
      .setSeed(42)

    val pipeline = new Pipeline().setStages(Array(assembler, rf))
    val model: PipelineModel = pipeline.fit(train)

    // ===== PREDICT =====
    val pred = model.transform(test)
      .select(col("prediction").cast("double"), col("label").cast("double"))
      .cache()

    // ===== METRICS =====
    val rdd = pred.rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    val metrics = new MulticlassMetrics(rdd)

    val cm = metrics.confusionMatrix
    val acc = metrics.accuracy

    val p0 = metrics.precision(0.0); val r0 = metrics.recall(0.0); val f0 = metrics.fMeasure(0.0)
    val p1 = metrics.precision(1.0); val r1 = metrics.recall(1.0); val f1 = metrics.fMeasure(1.0)

    val macroP = (p0 + p1) / 2.0
    val macroR = (r0 + r1) / 2.0
    val macroF = (f0 + f1) / 2.0

    println("\n================ TEST METRICS ================")
    println("Confusion Matrix (rows=true label, cols=pred):")
    println(cm)
    println(f"Accuracy     : $acc%.6f")
    println(f"Precision(0) : $p0%.6f   Recall(0): $r0%.6f   F1(0): $f0%.6f")
    println(f"Precision(1) : $p1%.6f   Recall(1): $r1%.6f   F1(1): $f1%.6f")
    println(f"Macro-Prec   : $macroP%.6f")
    println(f"Macro-Recall : $macroR%.6f")
    println(f"Macro-F1     : $macroF%.6f")

    // ===== SAVE MODEL =====
    // Overwrite safe
    model.write.overwrite().save(modelOut)
    println(s"[INFO] Saved RF model to: $modelOut")

    // ===== KEEP SPARK UI ALIVE =====
    // 10 minutes (change as you want)
    println("[INFO] Sleeping 10 minutes so you can view Spark UI ...")
    Thread.sleep(10 * 60 * 1000L)

    spark.stop()
  }
}