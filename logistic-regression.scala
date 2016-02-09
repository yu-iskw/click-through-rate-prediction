// ./bin/spark-shell --packages com.databricks:spark-csv_2.10:1.3.0
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, SQLContext, SaveMode}
import org.apache.spark.sql.types._

val trainParquetPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train.parquet/"
val testParquetPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test.parquet/"

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

// Loads training data and testing data
val train = sqlContext.read.parquet(trainParquetPath).cache()
val test = sqlContext.read.parquet(testParquetPath).cache()

val train4union = train.select(categoricalColumns.map(col):_*)
val test4union = test.select(categoricalColumns.map(col):_*)
val union = train4union.unionAll(test4union).cache()

// Extracts features
def getIndexedColumn(clm: String): String = s"${clm}_indexed"
def getColumnVec(clm: String): String = s"${clm}_vec"
val stages1 = ArrayBuffer.empty[PipelineStage]
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
val model1 = pipeline1.fit(union)
val trainDF = model1.transform(train).select('click, 'features).cache()
val testDF = model1.transform(test).select('id, 'features).cache()

// Exports data
// model1.transform(train).select($"label", $"features").
//   write.mode("overwrite").parquet(convertedTrainPath)
// model1.transform(test).select($"id", $"features").
//   write.mode("overwrite").parquet(convertedTestPath)

//val trainDF = model1.transform(train).select($"label", $"features")
//val testDF = model1.transform(test).select($"id", $"features")

// Reloads data
//val trainDF = sqlContext.read.parquet(convertedTrainPath).
//  repartition(160).cache()
//val testDF = sqlContext.read.parquet(convertedTestPath).cache()

// Trains a model
val strIdxrClick = new StringIndexer().setInputCol("click").setOutputCol("label")
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(strIdxrClick, lr))
val paramGrid = new ParamGridBuilder().
  addGrid(lr.labelCol, Array("label")).
  addGrid(lr.featuresCol, Array("features")).
  addGrid(lr.threshold, Array(0.22)).
  addGrid(lr.elasticNetParam, Array(0.0)).
  //addGrid(lr.fitIntercept, Array(false, true)).
  addGrid(lr.maxIter, Array(10)).
  build()
val cv = new CrossValidator().
  setEstimator(pipeline).
  setEvaluator(new BinaryClassificationEvaluator()).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(2)
val cvModel = cv.fit(trainDF)
cvModel.bestModel.parent match {
  case pipeline: Pipeline =>
    pipeline.getStages.zipWithIndex.foreach { case (stage, index) =>
      println(s"Stage[${index + 1}]: ${stage.getClass.getSimpleName}")
      println(stage.extractParamMap())
    }
}

val strIdxrClick = new StringIndexer().setInputCol("click").setOutputCol("label")
val lr2 = new LogisticRegression().setThreshold(0.22)
val cvModel= lr2.fit(strIdxrClick.fit(trainDF).transform(trainDF))

// Predicts with the trained model
val result0 = cvModel.transform(trainDF)
val result = cvModel.transform(testDF).select('id, 'probability).map {
  case Row(id: String, probability: Vector) => (id, probability(1))
}.toDF("id", "click").repartition(1)

val resultPath = "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/result/"
result.write.mode(SaveMode.Overwrite).
  format("com.databricks.spark.csv").
  option("header", "true").option("inferSchema", "true").
  save(resultPath)

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


val path = "s3n://s3-yu-ishikawa/test-data/amazon-fine-foods/Reviews.csv"
val data = sqlContext.read.format("com.databricks.spark.csv").
  option("header", "true").
  option("inferSchema", "true").
  load(path).cache()
import org.apache.spark.ml.feature._
val tokenizer = new Tokenizer().setInputCol("Text").setOutputCol("words")
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
val p = new Pipeline().setStages(Array(tokenizer, hashingTF))
val model = p.fit(data)
val data2 = model.transform(data)
