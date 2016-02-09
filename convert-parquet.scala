import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.{Row, SQLContext, SaveMode}
import org.apache.spark.sql.types._

val trainPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train"
val testPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test"

val trainParquetPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train.parquet/"
val testParquetPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test.parquet/"

val trainSchema = StructType(Array(
  StructField("id", LongType , true),
  StructField("click", IntegerType , true),
  StructField("hour", IntegerType , true),
  StructField("C1", IntegerType , true),
  StructField("banner_pos", IntegerType , true),
  StructField("site_id", StringType , true),
  StructField("site_domain", StringType , true),
  StructField("site_category", StringType , true),
  StructField("app_id", StringType , true),
  StructField("app_domain", StringType , true),
  StructField("app_category", StringType , true),
  StructField("device_id", StringType , true),
  StructField("device_ip", StringType , true),
  StructField("device_model", StringType , true),
  StructField("device_type", IntegerType , true),
  StructField("device_conn_type", IntegerType , true),
  StructField("C14", IntegerType , true),
  StructField("C15", IntegerType , true),
  StructField("C16", IntegerType , true),
  StructField("C17", IntegerType , true),
  StructField("C18", IntegerType , true),
  StructField("C19", IntegerType , true),
  StructField("C20", IntegerType , true),
  StructField("C21", IntegerType , true)
))

val testSchema = StructType(Array(
  StructField("id", LongType , true),
  StructField("hour", IntegerType , true),
  StructField("C1", IntegerType , true),
  StructField("banner_pos", IntegerType , true),
  StructField("site_id", StringType , true),
  StructField("site_domain", StringType , true),
  StructField("site_category", StringType , true),
  StructField("app_id", StringType , true),
  StructField("app_domain", StringType , true),
  StructField("app_category", StringType , true),
  StructField("device_id", StringType , true),
  StructField("device_ip", StringType , true),
  StructField("device_model", StringType , true),
  StructField("device_type", IntegerType , true),
  StructField("device_conn_type", IntegerType , true),
  StructField("C14", IntegerType , true),
  StructField("C15", IntegerType , true),
  StructField("C16", IntegerType , true),
  StructField("C17", IntegerType , true),
  StructField("C18", IntegerType , true),
  StructField("C19", IntegerType , true),
  StructField("C20", IntegerType , true),
  StructField("C21", IntegerType , true)
))

// Loads training data and testing data
val train = sqlContext.read.format("com.databricks.spark.csv").
  option("header", "true").option("inferSchema", "true").
  schema(trainSchema).load(trainPath).cache()
val test = sqlContext.read.format("com.databricks.spark.csv").
  option("header", "true").option("inferSchema", "true").
  schema(testSchema).load(testPath).cache()

train.write.mode("overwrite").parquet(trainParquetPath)
test.write.mode("overwrite").parquet(testParquetPath)
