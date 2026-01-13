import org.apache.spark.sql.{SparkSession, DataFrame, Column}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature._
import scala.collection.mutable.ArrayBuffer

/**
* Bot-IoT BULLETPROOF Preprocessor
*
* FIXES ALL SILENT KILLERS:
* 1. ✅ Read parquet ONCE (no re-reading for columns check)
* 2. ✅ NO full scan for counts - use FIXED sampling fraction
* 3. ✅ Use sampleBy() instead of filter+union (much faster)
* 4. ✅ DISK_ONLY persistence (no memory explosion)
* 5. ✅ Pipeline "keep" not "skip" (no silent row drops)
* 6. ✅ Fit pipeline on full sampled data (already small)
* 7. ✅ Add class_weight column for ML
*  
* Expected: 5-15 min runtime, NO crashes
*/
object BotIOTBulletproof {

case class Config(
inputPath: String = "hdfs://namenode:8020/datasets/bot_iot_parquet/merged",
outputBase: String = "hdfs://namenode:8020/datasets/bot_iot_parquet/preprocessed_fast",
attackSampleFrac: Double = 0.0013, // Fixed! Based on your counts: 9543*10/73360900
trainFrac: Double = 0.70,
valFrac: Double = 0.15,
testFrac: Double = 0.15,
seed: Long = 42L
)

def main(args: Array[String]): Unit = {
val spark = SparkSession.builder()
.appName("Bot-IoT BULLETPROOF")
.config("spark.sql.adaptive.enabled", "true")
.config("spark.sql.shuffle.partitions", "8") // Lower for small data
.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
.config("spark.sql.files.maxPartitionBytes", (256L * 1024 * 1024).toString)
.getOrCreate()

spark.sparkContext.setLogLevel("WARN")
import spark.implicits._

val cfg = if (args.length >= 2) {
Config(inputPath = args(0), outputBase = args(1))
} else Config()

val startTime = System.currentTimeMillis()

println("\n" + "="*80)
println("BULLETPROOF MODE: Fixed sampling, no double-reads, safe persistence")
println("="*80 + "\n")

// ========== STEP 1: READ ONCE + SAFE COLUMN SELECTION ==========
println("[1/5] Read parquet ONCE + safe column selection")

val rawCols = Seq("attack", "proto", "state", "dur", "mean", "stddev", "sum",  
"min", "max", "pkts", "bytes", "spkts", "dpkts", "sbytes",  
"dbytes", "rate", "srate", "drate", "sport", "dport", "flgs")

// Read schema ONCE (metadata only, no data scan)
val baseDF = spark.read.parquet(cfg.inputPath)
val availableCols = baseDF.columns.toSet
val safeCols = rawCols.filter(availableCols.contains)

println(s" → Available columns: ${safeCols.mkString(", ")}")

// Now read with projection + label
val raw = baseDF
.select(safeCols.map(col): _*)
.withColumn("label",  
when(coalesce(col("attack").cast(IntegerType), lit(1)) === 0, 0.0)
.otherwise(1.0)
)
.drop("attack")

// ========== STEP 2: STRATIFIED SAMPLE (NO FULL SCAN!) ==========
println(s"[2/5] Stratified sample (NO count scan!)")
println(s" → Normal: 100% (keep all ~9.5K)")
println(s" → Attack: ${cfg.attackSampleFrac * 100.0}% (sample ~95K from 73M)")

// ✅ sampleBy is WAY faster than filter+sample+union
val sampled = raw.stat.sampleBy(
"label",
Map(0.0 -> 1.0, 1.0 -> cfg.attackSampleFrac),
cfg.seed
).persist(StorageLevel.DISK_ONLY) // ✅ DISK_ONLY for safety

val sampledCount = sampled.count()
println(s" → Sampled: ~$sampledCount rows (vs 73M+ original)")
println(s" → Expected speedup: ~${73000000.0 / sampledCount.toDouble}x\n")

// ========== STEP 3: PROCESS SAMPLED DATA ==========
println("[3/5] Process sampled data (feature engineering)")
val processed = processSinglePass(sampled, spark)

// ========== STEP 4: BUILD PIPELINE (FIT ON FULL SAMPLED DATA) ==========
println("[4/5] Build ML pipeline")
val (finalDf, pipelineModel) = buildPipeline(processed, cfg.seed)

// Add class weights for ML models
val finalWithWeights = finalDf
.withColumn("class_weight",  
when(col("label") === 1.0, lit(10.0)) // Attack weight = ratio
.otherwise(lit(1.0))
)
.persist(StorageLevel.DISK_ONLY)

finalWithWeights.count() // materialize
println(s" → Final feature count: ${finalWithWeights.first().getAs[org.apache.spark.ml.linalg.Vector]("features").size}")

// ========== STEP 5: SPLIT & WRITE ==========
println("[5/5] Split & write")

val Array(train, valid, test) = finalWithWeights.randomSplit(
Array(cfg.trainFrac, cfg.valFrac, cfg.testFrac), cfg.seed
)

// Write with minimal partitions (data is small now)
train.coalesce(4).write.mode("overwrite")
.option("compression", "snappy")
.parquet(s"${cfg.outputBase}/train.parquet")
println(s" ✓ Train: ${train.count()} rows")

valid.coalesce(2).write.mode("overwrite")
.option("compression", "snappy")
.parquet(s"${cfg.outputBase}/val.parquet")
println(s" ✓ Val: ${valid.count()} rows")

test.coalesce(2).write.mode("overwrite")
.option("compression", "snappy")
.parquet(s"${cfg.outputBase}/test.parquet")
println(s" ✓ Test: ${test.count()} rows")

pipelineModel.write.overwrite().save(s"${cfg.outputBase}/pipelineModel")
println(s" ✓ Model saved")

println("\n" + "="*80)
println("FINAL DISTRIBUTION:")
finalWithWeights.groupBy($"label").count().orderBy($"label").show(false)

val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
println(f"TOTAL TIME: $elapsed%.1f seconds")
println("="*80 + "\n")

// Cleanup
finalWithWeights.unpersist()
sampled.unpersist()
spark.stop()
}

// ==================== PROCESS (SINGLE PASS) ====================
def processSinglePass(df: DataFrame, spark: SparkSession): DataFrame = {
import spark.implicits._
var out = df

// Proto bucket
if (out.columns.contains("proto")) {
val lc = lower(trim(col("proto")))
out = out.withColumn("proto_grp",
when(lc === "tcp" || lc === "udp", lc)
.when(lc.startsWith("icmp"), lit("icmp"))
.when(lc === "arp" || lc === "rarp", lit("arp"))
.otherwise(lit("other"))
).drop("proto")
}

// Port buckets
if (out.columns.contains("sport")) {
out = out.withColumn("sport_bucket",
when(col("sport").isNull || col("sport") < 0 || col("sport") > 65535, lit("invalid"))
.when(col("sport") <= 1023, lit("well_known"))
.when(col("sport") <= 49151, lit("registered"))
.otherwise(lit("dynamic"))
).drop("sport")
}

if (out.columns.contains("dport")) {
out = out.withColumn("dport_bucket",
when(col("dport").isNull || col("dport") < 0 || col("dport") > 65535, lit("invalid"))
.when(col("dport") <= 1023, lit("well_known"))
.when(col("dport") <= 49151, lit("registered"))
.otherwise(lit("dynamic"))
).drop("dport")
}

// Parse TCP flags (if present)
if (out.columns.contains("flgs")) {
out = out.withColumn("flgs_clean", regexp_replace(lower(trim(col("flgs"))), "0x", ""))
.withColumn("flgs_val",
when(col("flgs_clean").isNull || length(col("flgs_clean")) === 0, lit(0L))
.otherwise(expr("cast(conv(flgs_clean, 16, 10) as bigint)"))
)
.withColumn("syn", (col("flgs_val").bitwiseAND(lit(0x02L)) > 0).cast(DoubleType))
.withColumn("ack", (col("flgs_val").bitwiseAND(lit(0x10L)) > 0).cast(DoubleType))
.withColumn("fin", (col("flgs_val").bitwiseAND(lit(0x01L)) > 0).cast(DoubleType))
.withColumn("rst", (col("flgs_val").bitwiseAND(lit(0x04L)) > 0).cast(DoubleType))
.drop("flgs", "flgs_clean", "flgs_val")
} else {
// If no flags column, create zeros
out = out.withColumn("syn", lit(0.0))
.withColumn("ack", lit(0.0))
.withColumn("fin", lit(0.0))
.withColumn("rst", lit(0.0))
}

// Cast all numerics to double with coalesce to handle nulls
val numericCols = Seq("dur", "mean", "stddev", "sum", "min", "max",
"pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
"rate", "srate", "drate", "syn", "ack", "fin", "rst")

out = numericCols.filter(out.columns.contains).foldLeft(out) { (d, c) =>
d.withColumn(c, coalesce(col(c).cast(DoubleType), lit(0.0)))
}

// Feature engineering
def safeDiv(n: String, d: String): Column =  
when(col(d) > 0.0, col(n) / col(d)).otherwise(lit(0.0))

if (out.columns.contains("bytes") && out.columns.contains("pkts"))
out = out.withColumn("bytes_per_pkt", safeDiv("bytes", "pkts"))
else
out = out.withColumn("bytes_per_pkt", lit(0.0))

if (out.columns.contains("dur") && out.columns.contains("pkts"))
out = out.withColumn("pkts_per_sec", safeDiv("pkts", "dur"))
else
out = out.withColumn("pkts_per_sec", lit(0.0))

if (out.columns.contains("dur") && out.columns.contains("bytes"))
out = out.withColumn("bytes_per_sec", safeDiv("bytes", "dur"))
else
out = out.withColumn("bytes_per_sec", lit(0.0))

if (out.columns.contains("sbytes") && out.columns.contains("dbytes")) {
val denom = col("sbytes") + col("dbytes")
out = out.withColumn("byte_asymmetry",
when(denom > 0.0, (col("sbytes") - col("dbytes")) / denom).otherwise(lit(0.0))
)
} else {
out = out.withColumn("byte_asymmetry", lit(0.0))
}

if (out.columns.contains("spkts") && out.columns.contains("dpkts")) {
val denom = col("spkts") + col("dpkts")
out = out.withColumn("pkt_asymmetry",
when(denom > 0.0, (col("spkts") - col("dpkts")) / denom).otherwise(lit(0.0))
)
} else {
out = out.withColumn("pkt_asymmetry", lit(0.0))
}

if (out.columns.contains("srate") && out.columns.contains("drate")) {
out = out.withColumn("rate_diff", col("srate") - col("drate"))
.withColumn("rate_sum", col("srate") + col("drate"))
} else {
out = out.withColumn("rate_diff", lit(0.0))
.withColumn("rate_sum", lit(0.0))
}

// Log transform high-variance features
val logCandidates = Seq("dur", "pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
"rate", "srate", "drate", "bytes_per_pkt", "pkts_per_sec", "bytes_per_sec", "rate_sum")

out = logCandidates.filter(out.columns.contains).foldLeft(out) { (d, c) =>
d.withColumn(c, log1p(
when(col(c) < 0.0 || col(c).isNull, lit(0.0))
.when(col(c) > 1e12, lit(1e12))
.otherwise(col(c))
))
}

// Filter bad rows (NaN, Inf, etc)
val allNumeric = (numericCols ++ Seq("bytes_per_pkt", "pkts_per_sec", "bytes_per_sec",  
"byte_asymmetry", "pkt_asymmetry", "rate_diff", "rate_sum"))
.filter(out.columns.contains)

val validConds = allNumeric.flatMap { c =>
Seq(
col(c).isNotNull,
!isnan(col(c)),
col(c) =!= lit(Double.PositiveInfinity),
col(c) =!= lit(Double.NegativeInfinity)
)
}

if (validConds.nonEmpty) {
out = out.filter(validConds.reduce(_ && _))
}

// Keep only useful columns
val keep = Seq("label", "proto_grp", "state", "sport_bucket", "dport_bucket",
"dur", "mean", "stddev", "sum", "min", "max",
"pkts", "bytes", "spkts", "dpkts", "sbytes", "dbytes",
"rate", "srate", "drate", "syn", "ack", "fin", "rst",
"bytes_per_pkt", "pkts_per_sec", "bytes_per_sec",
"byte_asymmetry", "pkt_asymmetry", "rate_diff", "rate_sum"
).filter(out.columns.contains)

out.select(keep.map(col): _*)
}

// ==================== PIPELINE ====================
def buildPipeline(df: DataFrame, seed: Long): (DataFrame, PipelineModel) = {
val catCols = Seq("proto_grp", "state", "sport_bucket", "dport_bucket")
.filter(df.columns.contains)

val numericCols = df.schema.fields
.filter(f => f.dataType == DoubleType && f.name != "label")
.map(_.name)

val stages = ArrayBuffer.empty[PipelineStage]
val oheCols = ArrayBuffer.empty[String]

// String indexing + one-hot encoding for categoricals
catCols.foreach { c =>
val idx = new StringIndexer()
.setInputCol(c)
.setOutputCol(s"${c}_idx")
.setHandleInvalid("keep") // ✅ KEEP not SKIP!

val ohe = new OneHotEncoder()
.setInputCol(s"${c}_idx")
.setOutputCol(s"${c}_ohe")
.setDropLast(true)

stages += idx
stages += ohe
oheCols += s"${c}_ohe"
}

// Assemble + scale
val assembler = new VectorAssembler()
.setInputCols((numericCols ++ oheCols).toArray)
.setOutputCol("features_raw")
.setHandleInvalid("keep") // ✅ KEEP not SKIP!

val scaler = new StandardScaler()
.setInputCol("features_raw")
.setOutputCol("features")
.setWithMean(false)
.setWithStd(true)

stages += assembler
stages += scaler

val pipeline = new Pipeline().setStages(stages.toArray)

// ✅ Fit on FULL sampled dataset (already small ~100K rows)
println(s" → Fitting pipeline on full sampled data (~${df.count()} rows)")
val model = pipeline.fit(df)

// Transform
val out = model.transform(df).select(col("features"), col("label"))
(out, model)
}
}