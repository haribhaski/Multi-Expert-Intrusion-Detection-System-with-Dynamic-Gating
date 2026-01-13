import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector

object PreprocessNSLKDD {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("NSL-KDD Preprocessing v3 (Fixed)")
      .config("spark.sql.shuffle.partitions", "200")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    // ========================================
    // CONFIGURATION
    // ========================================
    val trainPath = if (args.length > 0) args(0) else "hdfs://namenode:8020/datasets/nsl-kdd/KDDTrain+.txt"
    val testPath  = if (args.length > 1) args(1) else "hdfs://namenode:8020/datasets/nsl-kdd/KDDTest+.txt"
    val outBase   = if (args.length > 2) args(2) else "hdfs://namenode:8020/datasets/processed/nsl-kdd"

    val pipelineOut = s"$outBase/pipeline"
    val trainOut    = s"$outBase/train_parquet"
    val testOut     = s"$outBase/test_parquet"

    println("=" * 80)
    println("NSL-KDD PREPROCESSING PIPELINE (v3 FIXED)")
    println("=" * 80)
    println(s"Train input: $trainPath")
    println(s"Test input:  $testPath")
    println(s"Output base: $outBase")
    println("=" * 80)

    // ========================================
    // NSL-KDD SCHEMA (43 columns total)
    // 41 features + label + difficulty
    // ========================================
    val colNames = Seq(
      "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
      "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
      "root_shell","su_attempted","num_root","num_file_creations","num_shells",
      "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
      "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
      "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
      "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
      "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
      "dst_host_rerror_rate","dst_host_srv_rerror_rate",
      "label","difficulty"
    )

    // ========================================
    // LOAD DATA
    // ========================================
    def loadNSLKDD(path: String): DataFrame = {
      spark.read
        .option("inferSchema", "false")
        .option("header", "false")
        .option("sep", ",")
        .csv(path)
        .toDF(colNames: _*)
    }

    println("\n[1/6] Loading datasets...")
    val rawTrain = loadNSLKDD(trainPath)
    val rawTest  = loadNSLKDD(testPath)

    // Cache early because we’ll count
    rawTrain.cache()
    rawTest.cache()

    val rawTrainN = rawTrain.count()
    val rawTestN  = rawTest.count()
    println(s"Raw train records: $rawTrainN")
    println(s"Raw test records:  $rawTestN")

    // ========================================
    // ATTACK NAME -> CATEGORY MAP (NSL-KDD)
    // ========================================
    // Categories: normal, dos, probe, r2l, u2r, unknown
    val dosAttacks = Seq("back","land","neptune","pod","smurf","teardrop",
      "apache2","udpstorm","processtable","worm")
    val probeAttacks = Seq("satan","ipsweep","nmap","portsweep","mscan","saint")
    val r2lAttacks = Seq("guess_passwd","ftp_write","imap","phf","multihop","warezmaster",
      "warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel",
      "sendmail","named")
    val u2rAttacks = Seq("buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps")

    // ========================================
    // DATA CLEANING
    // ========================================
    println("\n[2/6] Cleaning data...")

    def cleanData(df: DataFrame): DataFrame = {
      val cleaned = df
        .withColumn("label_raw", lower(regexp_replace(col("label"), "\\.$", "")))
        // Binary label: normal=0, attack=1
        .withColumn("label_binary", when(col("label_raw") === "normal", 0.0).otherwise(1.0))
        // Proper NSL-KDD category mapping
        .withColumn("attack_category",
          when(col("label_raw") === "normal", "normal")
            .when(col("label_raw").isin(dosAttacks: _*), "dos")
            .when(col("label_raw").isin(probeAttacks: _*), "probe")
            .when(col("label_raw").isin(r2lAttacks: _*), "r2l")
            .when(col("label_raw").isin(u2rAttacks: _*), "u2r")
            .otherwise("unknown")
        )

      cleaned
    }

    val trainCleaned = cleanData(rawTrain)
    val testCleaned  = cleanData(rawTest)

    // ========================================
    // FEATURE ENGINEERING
    // ========================================
    println("\n[3/6] Engineering features...")

    val categoricalCols = Array("protocol_type", "service", "flag")

    val numericCols = colNames
      .filterNot(categoricalCols.contains)
      .filterNot(c => c == "label" || c == "difficulty")
      .toArray

    def castNumeric(df: DataFrame): DataFrame =
      numericCols.foldLeft(df) { (acc, c) => acc.withColumn(c, col(c).cast("double")) }

    val trainTyped = castNumeric(trainCleaned)
    val testTyped  = castNumeric(testCleaned)

    val trainFiltered = trainTyped
      .filter(col("duration").isNotNull)
      .filter(col("src_bytes").isNotNull && col("src_bytes") >= 0)
      .filter(col("dst_bytes").isNotNull && col("dst_bytes") >= 0)
      .na.fill(0.0, numericCols)
      .cache()

    val testFiltered = testTyped
      .filter(col("duration").isNotNull)
      .filter(col("src_bytes").isNotNull && col("src_bytes") >= 0)
      .filter(col("dst_bytes").isNotNull && col("dst_bytes") >= 0)
      .na.fill(0.0, numericCols)
      .cache()

    val trainN = trainFiltered.count()
    val testN  = testFiltered.count()
    println(s"After cleaning - Train: $trainN, Test: $testN")

    // ========================================
    // BUILD ML PIPELINE
    // ========================================
    println("\n[4/6] Building ML pipeline...")

    val indexers = categoricalCols.map { c =>
      new StringIndexer()
        .setInputCol(c)
        .setOutputCol(s"${c}_idx")
        .setHandleInvalid("keep")
    }

    val encoder = new OneHotEncoder()
      .setInputCols(categoricalCols.map(_ + "_idx"))
      .setOutputCols(categoricalCols.map(_ + "_vec"))
      .setDropLast(false)
      .setHandleInvalid("keep")

    val featureCols = numericCols ++ categoricalCols.map(_ + "_vec")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")
      .setHandleInvalid("keep")   // IMPORTANT: don't silently drop rows

    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithMean(false)         // IMPORTANT: keep sparse, avoid memory blow-up
      .setWithStd(true)

    val pipeline = new Pipeline().setStages(indexers ++ Array(encoder, assembler, scaler))

    // ========================================
    // FIT PIPELINE ON TRAINING DATA ONLY
    // ========================================
    println("\n[5/6] Fitting pipeline on training data...")
    val pipelineModel: PipelineModel = pipeline.fit(trainFiltered)

    val trainTransformed = pipelineModel.transform(trainFiltered)
      .select(
        col("label_binary").alias("label"),
        col("attack_category"),
        col("label_raw"),
        col("features")
      )
      .cache()

    val testTransformed = pipelineModel.transform(testFiltered)
      .select(
        col("label_binary").alias("label"),
        col("attack_category"),
        col("label_raw"),
        col("features")
      )
      .cache()

    // Materialize caches once
    val trainTN = trainTransformed.count()
    val testTN  = testTransformed.count()

    // ========================================
    // SAVE OUTPUTS
    // ========================================
    println("\n[6/6] Saving outputs...")

    pipelineModel.write.overwrite().save(pipelineOut)
    trainTransformed.write.mode("overwrite").parquet(trainOut)
    testTransformed.write.mode("overwrite").parquet(testOut)

    println(s"✓ Pipeline saved to:      $pipelineOut")
    println(s"✓ Train parquet saved to: $trainOut  (rows=$trainTN)")
    println(s"✓ Test parquet saved to:  $testOut   (rows=$testTN)")

    // ========================================
    // QUICK STATS
    // ========================================
    println("\n" + "=" * 80)
    println("DATASET STATISTICS")
    println("=" * 80)

    println("\nTrain label distribution:")
    trainTransformed.groupBy("label").count().show(false)

    println("\nTrain category distribution:")
    trainTransformed.groupBy("attack_category").count().orderBy(desc("count")).show(false)

    println("\nTest label distribution:")
    testTransformed.groupBy("label").count().show(false)

    println("\nTest category distribution:")
    testTransformed.groupBy("attack_category").count().orderBy(desc("count")).show(false)

    val featureSize = trainTransformed
    .select("features")
    .head()
    .getAs[Vector](0)
    .size

  println(s"\nFeature vector dimension: $featureSize")

    println("\nPREPROCESSING COMPLETE!")
    spark.stop()
  }
}
