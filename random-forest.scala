// ./bin/spark-shell --packages com.databricks:spark-csv_2.10:1.3.0
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{SQLContext, SaveMode}

val trainPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train/"
val testPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test"
val convertedTrainPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train.converted.parquet/"
val convertedTestPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test.converted.parquet/"

val categoricalColumns = Array(
  "banner_pos",
  "site_id", "site_domain", "site_category",
  "app_domain", "app_category",
  "device_model", "device_type", "device_conn_type",
  "C1", "C14", "C15", "C16", "C17",
  "C18", "C19", "C20", "C21"
)
//val categoricalColumns = Array(
//  "banner_pos",
//  "site_category","app_category",
//  "C1", "C14", "C15", "C16", "C17",
//  "C18", "C19", "C20", "C21"
//)

// Loads training data and testing data
val train = sqlContext.read.format("com.databricks.spark.csv").
  option("header", "true").option("inferSchema", "true").
  load(trainPath).cache()
val test = sqlContext.read.format("com.databricks.spark.csv").
  option("header", "true").option("inferSchema", "true").
  load(testPath).cache()

// Formats data
def getIndexedColumn(clm: String): String = s"${clm}_indexed"
def getColumnVec(clm: String): String = s"${clm}_vec"
val stages1 = ArrayBuffer.empty[PipelineStage]
val strIdxrClick = new StringIndexer().
  setInputCol("click").
  setOutputCol("label")
  stages1.append(strIdxrClick)
categoricalColumns.foreach { clm =>
  val stringIndexer = new StringIndexer().
    setInputCol(clm).
    setOutputCol(getIndexedColumn(clm)).
    setHandleInvalid("skip")
  val oneHotEncoder = new OneHotEncoder().
    setInputCol(getIndexedColumn(clm)).
    setOutputCol(getColumnVec(clm))
  Array(stringIndexer, oneHotEncoder)
  stages1.append(stringIndexer)
  stages1.append(oneHotEncoder)
}
val va = new VectorAssembler().
  setInputCols(categoricalColumns.map(getColumnVec)).
  setOutputCol("features")
  stages1.append(va)
val pipeline1 = new Pipeline().setStages(stages1.toArray)
val model1 = pipeline1.fit(train)
model1.transform(train).select($"label", $"features").
  write.mode("overwrite").parquet(convertedTrainPath)
model1.transform(test).select($"id", $"features").
  write.mode("overwrite").parquet(convertedTestPath)


val trainDF = sqlContext.read.parquet(convertedTrainPath).
  repartition(1280).sample(false, 0.5).cache()
  //repartition(640).sample(false, 0.1).cache()
val testDF = sqlContext.read.parquet(convertedTestPath).cache()

// Trains a model
val rf = new RandomForestClassifier()
val pipeline = new Pipeline().setStages(Array(rf))
val paramGrid = new ParamGridBuilder().
  addGrid(rf.labelCol, Array("label")).
  addGrid(rf.featuresCol, Array("features")).
  //addGrid(rf.featureSubsetStrategy, Array("log2")).
  // addGrid(rf.subsamplingRate, Array(0.4, 0.8, 1.0)).
  // addGrid(rf.maxBins, Array(16, 32, 48)).
  // addGrid(rf.maxDepth, Array(3, 5, 7)).
  // addGrid(rf.numTrees, Array(10, 20, 30)). 
  addGrid(rf.cacheNodeIds, Array(true)).
  build()
val cv = new CrossValidator().
  setEstimator(pipeline).
  setEvaluator(new BinaryClassificationEvaluator()).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(3)
val cvModel = cv.fit(trainDF)
cvModel.bestModel.parent match {
  case pipeline: Pipeline =>
    pipeline.getStages.zipWithIndex.foreach { case (stage, index) =>
      println(s"Stage[${index + 1}]: ${stage.getClass.getSimpleName}")
      println(stage.extractParamMap())
    }
}

// Predicts with the trained model
val result0 = cvModel.transform(trainDF)
