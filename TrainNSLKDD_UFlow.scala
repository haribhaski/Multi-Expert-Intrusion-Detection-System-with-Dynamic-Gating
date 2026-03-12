import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.storage.StorageLevel

object TrainNSLKDD_UFlow_RF {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("NSLKDD_UFlow_RF")
      .getOrCreate()

    val inPath   = "hdfs://namenode:8020/datasets/processed/uflow/nslkdd/uflow_20k_v1"
    val modelOut = "hdfs://namenode:8020/models/experts/nslkdd/rf_uflow_v1"

    val df = spark.read.parquet(inPath)

    val featureCols = Array(
      "duration_s","src_bytes","dst_bytes",
      "total_bytes","bytes_ratio","bytes_per_sec",
      "log_duration","log_src_bytes",
      "proto_tcp","proto_udp","proto_icmp"
    )

    // Add stable row id FIRST so split stays identical across plans
    val clean = df.select(
        (featureCols.map(c => col(c).cast("double").as(c)) :+ col("label").cast("double").as("label")): _*
      )
      .na.fill(0.0)
      .withColumn("rid", monotonically_increasing_id())
      .persist(StorageLevel.MEMORY_AND_DISK)

    // ---- Your original accuracy behavior: assemble then split
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val finalDf = assembler.transform(clean).select("rid","features","label")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val Array(trainF, testF) = finalDf.randomSplit(Array(0.8, 0.2), seed = 42)
    val train = trainF.persist(StorageLevel.MEMORY_AND_DISK)
    val test  = testF.persist(StorageLevel.MEMORY_AND_DISK)

    println(s"train=${train.count()}, test=${test.count()}")

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(300)
      .setMaxDepth(12)
      .setFeatureSubsetStrategy("log2")
      .setSubsamplingRate(0.8)
      .setSeed(42)
      .setMaxBins(128)

    // Train EXACTLY like your best code
    val rfModel = rf.fit(train)

    // Evaluate (same)
    val pred = rfModel.transform(test).cache()

    val predictionAndLabels = pred.select(col("prediction"), col("label"))
      .na.drop()
      .rdd.map(r => (r.getDouble(0), r.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("\n================ EVALUATION (TEST) ================")
    println("Confusion Matrix (rows=true, cols=pred):")
    println(metrics.confusionMatrix)

    println(f"Accuracy      : ${metrics.accuracy}%.6f")
    println(f"Weighted F1   : ${metrics.weightedFMeasure}%.6f")
    println(f"Weighted Prec : ${metrics.weightedPrecision}%.6f")
    println(f"Weighted Rec  : ${metrics.weightedRecall}%.6f")

    val auc = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
      .evaluate(pred)

    println(f"AUC (ROC)     : $auc%.6f")

    // ---- Now save as PipelineModel using the SAME train/test rows (by rid)
    val trainIds = train.select("rid")
    val testIds  = test.select("rid")

    val trainRaw = clean.join(trainIds, Seq("rid"), "inner").drop("rid")
    val testRaw  = clean.join(testIds,  Seq("rid"), "inner").drop("rid")

    val pipeline = new Pipeline().setStages(Array(assembler, rf))
    val pipelineModel: PipelineModel = pipeline.fit(trainRaw)

    // optional sanity: pipeline preds should match close
    val _ = pipelineModel.transform(testRaw).count()

    pipelineModel.write.overwrite().save(modelOut)
    println("\nSaved PIPELINE model (accuracy-preserving split): " + modelOut)

    Thread.sleep(60000)

    pred.unpersist()
    train.unpersist(); test.unpersist()
    finalDf.unpersist(); clean.unpersist()
    spark.stop()
  }
}