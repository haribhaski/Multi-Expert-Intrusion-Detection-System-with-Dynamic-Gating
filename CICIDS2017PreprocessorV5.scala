import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.hadoop.fs.{FileSystem, Path => HPath}

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object CICIDS2017PreprocessorV5 {

  case class CleanStats(
    rawRows: Long,
    afterDedupRows: Long,
    afterLeakageDropRows: Long,
    afterCastRows: Long,
    afterLabelFilterRows: Long
  )

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("CICIDS2017 Preprocessor V5")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val inputGlob         = argOr(args, 0, "hdfs://namenode:8020/datasets/cicids2017/*.csv")
    val outputBase        = argOr(args, 1, "hdfs://namenode:8020/datasets/processed/cicids2017_v5")
    val imputeStr         = argOr(args, 2, "median").trim.toLowerCase
    val sampleFrac        = argOr(args, 3, "1.0").toDouble
    val cacheDuringAudit  = argOr(args, 4, "true").toBoolean
    val enableLogTf       = argOr(args, 5, "false").toBoolean
    val skewThreshold     = argOr(args, 6, "5.0").toDouble
    val outputPartitions  = argOr(args, 7, "0").toInt
    val trainFrac         = argOr(args, 8, "0.70").toDouble
    val valFrac           = argOr(args, 9, "0.15").toDouble
    val testFrac          = argOr(args, 10, "0.15").toDouble
    val savePipelineModel = argOr(args, 11, "true").toBoolean

    require(sampleFrac > 0.0 && sampleFrac <= 1.0, s"sampleFrac must be in (0,1], got $sampleFrac")
    require(imputeStr == "median" || imputeStr == "mean", s"imputeStrategy must be median|mean, got $imputeStr")
    require(math.abs((trainFrac + valFrac + testFrac) - 1.0) < 1e-9, "train/val/test fractions must sum to 1.0")
    require(outputPartitions >= 0, s"outputPartitions must be >= 0, got $outputPartitions")

    // 1) Load all CSVs safely (per-file -> normalize -> unionByName)
    val rawUnion = loadCsvUnionSchemaSafe(spark, inputGlob)

    // 2) Optional sampling (after union)
    val raw = if (sampleFrac < 1.0) rawUnion.sample(withReplacement = false, fraction = sampleFrac, seed = 42L) else rawUnion

    // 3) Clean + cast
    val (cleaned, stats, cachedToUnpersist) = cleanAndCast(raw, cacheDuringAudit)

    // 4) Feature engineering
    var df = addFeatures(cleaned)

    // 5) Impute
    val (imputed, imputedCols) =
      if (imputeStr == "median") imputeMedianSinglePass(df, labelCol = "Label")
      else imputeMeanSinglePass(df, labelCol = "Label")
    df = imputed

    // 6) Optional log transform (skew proxy)
    if (enableLogTf) {
      df = logTransformSkewed(df, labelCol = "Label", skewThreshold = skewThreshold)
    }

    // 7) Low-variance drop
    val (reduced, lowVarDropped) = dropLowVariance(df, labelCol = "Label", threshold = 1e-4)

    // 8) Assemble + scale
    val (finalDf, pipelineModel) = assembleAndScale(reduced)

    // 9) Partition control and cache for split + counts
    val toWrite0 = if (outputPartitions > 0) finalDf.repartition(outputPartitions) else finalDf
    val toWrite  = toWrite0.persist(StorageLevel.MEMORY_AND_DISK)
    val finalCount = toWrite.count()

    // 10) Write outputs
    toWrite.write.mode("overwrite").parquet(s"$outputBase/all.parquet")

    val (trainDf, valDf, testDf) = splitRandom(toWrite, trainFrac, valFrac, testFrac, seed = 42L)
    trainDf.write.mode("overwrite").parquet(s"$outputBase/train.parquet")
    valDf.write.mode("overwrite").parquet(s"$outputBase/val.parquet")
    testDf.write.mode("overwrite").parquet(s"$outputBase/test.parquet")

    if (savePipelineModel) {
      pipelineModel.write.overwrite().save(s"$outputBase/pipelineModel")
    }

    // 11) Audit
    println("\n=== CICIDS2017 PREPROCESS AUDIT (V5) ===")
    println(s"Input glob:           $inputGlob")
    println(s"Output base:          $outputBase")
    println(s"Sampling fraction:    $sampleFrac")
    println(s"Imputation:           $imputeStr (cols imputed: $imputedCols)")
    println(s"Log transform:        $enableLogTf (skewThreshold=$skewThreshold)")
    println(s"Low-var cols dropped: ${lowVarDropped.size}")
    if (lowVarDropped.nonEmpty) println(s"  Low-var drop sample: ${lowVarDropped.take(25).mkString(", ")}${if (lowVarDropped.size > 25) " ..." else ""}")
    println(s"Output partitions:    $outputPartitions (0 means unchanged)")
    println(s"Split:                train=$trainFrac, val=$valFrac, test=$testFrac")
    println(s"Save pipeline model:  $savePipelineModel")

    println("\nRow counts:")
    println(s"  raw(unioned+sampled) = ${stats.rawRows}")
    println(s"  after dedup          = ${stats.afterDedupRows}")
    println(s"  after leakage drop   = ${stats.afterLeakageDropRows}")
    println(s"  after cast           = ${stats.afterCastRows}")
    println(s"  after label filter   = ${stats.afterLabelFilterRows}")
    println(s"  final(all)           = $finalCount")

    println("\nLabel distribution (all):")
    toWrite.groupBy("Label", "is_attack").count().orderBy(desc("count")).show(50, truncate = false)

    // Cleanup caches
    cachedToUnpersist.foreach(_.unpersist(blocking = false))
    toWrite.unpersist(blocking = false)

    spark.stop()
  }

  // ---------------- Helpers ----------------

  private def argOr(args: Array[String], idx: Int, default: String): String =
    if (args.length > idx) args(idx) else default

  private def maybeCache(df: DataFrame, enable: Boolean): DataFrame =
    if (!enable) df else df.persist(StorageLevel.MEMORY_AND_DISK)

  // ---------------- Load: schema-safe union ----------------

  /**
   * Expands the input glob (HDFS) to concrete files, reads each CSV separately,
   * trims and de-duplicates headers deterministically, then unions by name.
   */
  def loadCsvUnionSchemaSafe(spark: SparkSession, inputGlob: String): DataFrame = {
    val conf = spark.sparkContext.hadoopConfiguration

    // Determine FS from URI (works for hdfs://namenode:8020/..., file:/..., etc.)
    val uri = new java.net.URI(inputGlob)
    val fs = FileSystem.get(uri, conf)

    val statuses = fs.globStatus(new HPath(inputGlob))
    require(statuses != null && statuses.nonEmpty, s"No files matched input glob: $inputGlob")

    val paths = statuses.map(_.getPath.toString).toSeq.sorted
    println(s"Found ${paths.size} CSV files:")
    paths.foreach(p => println(s"  - $p"))

    def normalizeCols(df: DataFrame): DataFrame = {
      // Trim first
      val trimmed = df.columns.foldLeft(df) { (d, c) =>
        val t = c.trim
        if (t != c) d.withColumnRenamed(c, t) else d
      }

      // Deterministic de-dup
      val seen = scala.collection.mutable.Map[String, Int]()
      val renamePairs = trimmed.columns.toSeq.map { c =>
        val n = seen.getOrElse(c, 0) + 1
        seen.update(c, n)
        if (n == 1) (c, c) else (c, s"${c}__dup$n")
      }

      renamePairs.foldLeft(trimmed) { case (d, (from, to)) =>
        if (from == to) d else d.withColumnRenamed(from, to)
      }
    }

    val dfs = paths.map { p =>
      val df = spark.read
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("multiLine", "false")
        .csv(p)

      normalizeCols(df)
    }

    dfs.reduce((a, b) => a.unionByName(b, allowMissingColumns = true))
  }

  // ---------------- Clean & Cast ----------------

  /**
   * Trims/normalizes Label column name (case-insensitive),
   * drops duplicates,
   * drops leakage columns (including dup variants),
   * casts features to Double,
   * replaces NaN/Inf with null,
   * filters Label != null.
   *
   * Returns:
   *  - cleaned DF
   *  - stats
   *  - cached DFs to unpersist (if caching enabled)
   */
  def cleanAndCast(df0: DataFrame, cacheDuringAudit: Boolean): (DataFrame, CleanStats, Seq[DataFrame]) = {
    val rawRows = df0.count()

    // Normalize Label column if needed (case-insensitive)
    val labelNameOpt = df0.columns.find(_.equalsIgnoreCase("Label"))
    val df1 = labelNameOpt match {
      case Some(name) if name != "Label" => df0.withColumnRenamed(name, "Label")
      case _ => df0
    }

    val df2 = maybeCache(df1.dropDuplicates(), cacheDuringAudit)
    val afterDedupRows = df2.count()

    // Drop leakage columns (and their __dup variants)
    val leakageBase = Seq(
      "Flow ID",
      "Src IP", "Source IP",
      "Dst IP", "Destination IP",
      "Src Port", "Source Port",
      "Dst Port", "Destination Port",
      "Timestamp"
    )

    val leakageToDrop = df2.columns.filter { c =>
      leakageBase.exists(base => c == base || c.startsWith(base + "__dup"))
    }

    val df3 = maybeCache(if (leakageToDrop.nonEmpty) df2.drop(leakageToDrop: _*) else df2, cacheDuringAudit)
    val afterLeakageDropRows = df3.count()

    // Cast everything except Label to Double (strings -> Double, failures -> null)
    val featureCandidates = df3.columns.filter(_ != "Label")
    val casted = featureCandidates.foldLeft(df3) { (d, c) =>
      d.withColumn(c, regexp_replace(col(c), ",", "").cast(DoubleType))
    }

    // NaN/Inf -> null
    val fixed = featureCandidates.foldLeft(casted) { (d, c) =>
      d.withColumn(
        c,
        when(col(c).isNull, lit(null).cast(DoubleType))
          .when(isnan(col(c)), lit(null).cast(DoubleType))
          .when(col(c) === lit(Double.PositiveInfinity), lit(null).cast(DoubleType))
          .when(col(c) === lit(Double.NegativeInfinity), lit(null).cast(DoubleType))
          .otherwise(col(c))
      )
    }

    val df4 = maybeCache(fixed, cacheDuringAudit)
    val afterCastRows = df4.count()

    val out = maybeCache(df4.filter(col("Label").isNotNull), cacheDuringAudit)
    val afterLabelFilterRows = out.count()

    val stats = CleanStats(
      rawRows = rawRows,
      afterDedupRows = afterDedupRows,
      afterLeakageDropRows = afterLeakageDropRows,
      afterCastRows = afterCastRows,
      afterLabelFilterRows = afterLabelFilterRows
    )

    val cached = if (cacheDuringAudit) Seq(df2, df3, df4, out).distinct else Seq.empty
    (out, stats, cached)
  }

  // ---------------- Feature Engineering ----------------

  def addFeatures(df: DataFrame): DataFrame = {
    def has(cols: String*): Boolean = cols.forall(df.columns.contains)
    var out = df

    // Flow Duration (microseconds) -> seconds
    if (out.columns.contains("Flow Duration")) {
      out = out.withColumn(
        "flow_duration_sec",
        when(col("Flow Duration") > 0.0, col("Flow Duration") / 1e6).otherwise(0.0)
      )
    }

    if (has("Total Fwd Packets", "Total Backward Packets") && out.columns.contains("flow_duration_sec")) {
      out = out.withColumn(
        "packets_per_sec",
        when(
          col("flow_duration_sec") > 0.0,
          (col("Total Fwd Packets") + col("Total Backward Packets")) / col("flow_duration_sec")
        ).otherwise(0.0)
      )
    }

    if (has("Total Length of Fwd Packets", "Total Length of Bwd Packets", "Total Fwd Packets", "Total Backward Packets")) {
      out = out.withColumn(
        "bytes_per_packet",
        when(
          (col("Total Fwd Packets") + col("Total Backward Packets")) > 0.0,
          (col("Total Length of Fwd Packets") + col("Total Length of Bwd Packets")) /
            (col("Total Fwd Packets") + col("Total Backward Packets"))
        ).otherwise(0.0)
      )
    }

    // Binary label for cross-dataset generalization
    out.withColumn("is_attack", when(col("Label") === "BENIGN", 0.0).otherwise(1.0))
  }

  // ---------------- Imputation (single pass) ----------------

  def imputeMedianSinglePass(df: DataFrame, labelCol: String): (DataFrame, Int) = {
    val numericCols = df.schema.fields
      .filter(f => f.name != labelCol && f.dataType == DoubleType && f.name != "is_attack")
      .map(_.name)

    if (numericCols.isEmpty) return (df, 0)

    val accuracy = 10000
    val exprs = numericCols.map { c =>
      expr(s"percentile_approx(`${c}`, 0.5, $accuracy)").as(c)
    }

    val row = df.agg(exprs.head, exprs.tail: _*).head()

    val fillMap = numericCols.map { c =>
      val v = row.getAs[Double](c)
      c -> (if (v.isNaN) 0.0 else v)
    }.toMap

    (df.na.fill(fillMap), fillMap.size)
  }

  def imputeMeanSinglePass(df: DataFrame, labelCol: String): (DataFrame, Int) = {
    val numericCols = df.schema.fields
      .filter(f => f.name != labelCol && f.dataType == DoubleType && f.name != "is_attack")
      .map(_.name)

    if (numericCols.isEmpty) return (df, 0)

    val exprs = numericCols.map(c => avg(col(c)).as(c))
    val row = df.agg(exprs.head, exprs.tail: _*).head()

    val fillMap = numericCols.map { c =>
      val v = row.getAs[Double](c)
      c -> (if (v.isNaN) 0.0 else v)
    }.toMap

    (df.na.fill(fillMap), fillMap.size)
  }

  // ---------------- Optional log transform (skew proxy) ----------------

  /**
   * Fast skew proxy: (mean - median) / std.
   * If abs(proxy) > skewThreshold -> log1p transform.
   * One aggregation for all columns.
   */
  def logTransformSkewed(df: DataFrame, labelCol: String, skewThreshold: Double): DataFrame = {
    val numericCols = df.schema.fields
      .filter(f => f.name != labelCol && f.dataType == DoubleType && f.name != "is_attack")
      .map(_.name)

    if (numericCols.isEmpty) return df

    val acc = 5000
    val meanExprs = numericCols.map(c => avg(col(c)).as(s"${c}__mean"))
    val stdExprs  = numericCols.map(c => stddev_pop(col(c)).as(s"${c}__std"))
    val medExprs  = numericCols.map(c => expr(s"percentile_approx(`${c}`, 0.5, $acc)").as(s"${c}__med"))

    val allExprs = meanExprs ++ stdExprs ++ medExprs
    val row = df.agg(allExprs.head, allExprs.tail: _*).head()

    val toLog = numericCols.filter { c =>
      val mu  = row.getAs[Double](s"${c}__mean")
      val sd  = row.getAs[Double](s"${c}__std")
      val med = row.getAs[Double](s"${c}__med")

      val proxy =
        if (sd == 0.0 || sd.isNaN) 0.0
        else (mu - med) / sd

      math.abs(proxy) > skewThreshold
    }

    if (toLog.nonEmpty) {
      println(s"Log1p transform columns (count=${toLog.size}): ${toLog.take(30).mkString(", ")}${if (toLog.size > 30) " ..." else ""}")
    }

    toLog.foldLeft(df) { (d, c) =>
      d.withColumn(c, log1p(when(col(c) < 0.0, 0.0).otherwise(col(c))))
    }
  }

  // ---------------- Low variance drop ----------------

  def dropLowVariance(df: DataFrame, labelCol: String, threshold: Double): (DataFrame, Seq[String]) = {
    val cols = df.schema.fields
      .filter(f => f.name != labelCol && f.dataType == DoubleType && f.name != "is_attack")
      .map(_.name)

    if (cols.isEmpty) return (df, Seq.empty)

    val exprs = cols.map(c => variance(col(c)).as(c))
    val row = df.agg(exprs.head, exprs.tail: _*).head()

    val lowVar = cols.filter { c =>
      val v = row.getAs[Double](c)
      v.isNaN || v < threshold
    }

    val out = if (lowVar.nonEmpty) df.drop(lowVar: _*) else df
    (out, lowVar)
  }

  // ---------------- Assemble & scale ----------------

  def assembleAndScale(df: DataFrame): (DataFrame, PipelineModel) = {
    val featureCols = df.schema.fields
      .filter(f => f.dataType == DoubleType && f.name != "is_attack")
      .map(_.name)

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features_raw")
      .setHandleInvalid("skip")

    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false) // important: avoid densifying

    val pipeline = new Pipeline().setStages(Array(assembler, scaler))
    val model = pipeline.fit(df)

    val out = model.transform(df).select(col("features"), col("is_attack"), col("Label"))
    (out, model)
  }

  // ---------------- Split ----------------

  def splitRandom(df: DataFrame, trainFrac: Double, valFrac: Double, testFrac: Double, seed: Long): (DataFrame, DataFrame, DataFrame) = {
    val Array(train, valid, test) = df.randomSplit(Array(trainFrac, valFrac, testFrac), seed)
    (train, valid, test)
  }
}
