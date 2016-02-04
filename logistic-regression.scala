// ./bin/spark-shell --packages com.databricks:spark-csv_2.10:1.3.0
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
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
// Exports data
val model1 = pipeline1.fit(train)
model1.transform(train).select($"label", $"features").
  write.mode("overwrite").parquet(convertedTrainPath)
model1.transform(test).select($"id", $"features").
  write.mode("overwrite").parquet(convertedTestPath)

// Reloads data
val trainDF = sqlContext.read.parquet(convertedTrainPath).
  repartition(640).cache()
  repartition(640).sample(false, 0.5).cache()
val testDF = sqlContext.read.parquet(convertedTestPath).cache()

// Trains a model
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(lr))
val paramGrid = new ParamGridBuilder().
  addGrid(lr.labelCol, Array("label")).
  addGrid(lr.featuresCol, Array("features")).
  addGrid(lr.threshold, Array(0.25, 0.5, 0.75)).
  addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).
  // addGrid(lr.fitIntercept, Array(false, true)).
  addGrid(lr.maxIter, Array(50, 100)).
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

// Stage[1]: LogisticRegression
// {
//   logreg_006eb1862908-elasticNetParam: 0.5,
//   logreg_006eb1862908-featuresCol: features,
//   logreg_006eb1862908-fitIntercept: true,
//   logreg_006eb1862908-labelCol: label,
//   logreg_006eb1862908-maxIter: 100,
//   logreg_006eb1862908-predictionCol: prediction,
//   logreg_006eb1862908-probabilityCol: probability,
//   logreg_006eb1862908-rawPredictionCol: rawPrediction,
//   logreg_006eb1862908-regParam: 0.0,
//   logreg_006eb1862908-standardization: true,
//   logreg_006eb1862908-threshold: 0.25,
//   logreg_006eb1862908-tol: 1.0E-6,
//   logreg_006eb1862908-weightCol:
// 
// }
