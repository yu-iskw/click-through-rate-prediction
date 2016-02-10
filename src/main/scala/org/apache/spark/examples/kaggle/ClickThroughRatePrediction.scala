/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.kaggle

import scala.collection.mutable.ArrayBuffer

import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, SQLContext, SaveMode}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

// scalastyle:off println
object ClickThroughRatePrediction {

  private val trainSchema = StructType(Array(
    StructField("id", StringType, false),
    StructField("click", IntegerType, true),
    StructField("hour", IntegerType, true),
    StructField("C1", IntegerType, true),
    StructField("banner_pos", IntegerType, true),
    StructField("site_id", StringType, true),
    StructField("site_domain", StringType, true),
    StructField("site_category", StringType, true),
    StructField("app_id", StringType, true),
    StructField("app_domain", StringType, true),
    StructField("app_category", StringType, true),
    StructField("device_id", StringType, true),
    StructField("device_ip", StringType, true),
    StructField("device_model", StringType, true),
    StructField("device_type", IntegerType, true),
    StructField("device_conn_type", IntegerType, true),
    StructField("C14", IntegerType, true),
    StructField("C15", IntegerType, true),
    StructField("C16", IntegerType, true),
    StructField("C17", IntegerType, true),
    StructField("C18", IntegerType, true),
    StructField("C19", IntegerType, true),
    StructField("C20", IntegerType, true),
    StructField("C21", IntegerType, true)
  ))

  private val testSchema = StructType(Array(
    StructField("id", StringType, false),
    StructField("hour", IntegerType, true),
    StructField("C1", IntegerType, true),
    StructField("banner_pos", IntegerType, true),
    StructField("site_id", StringType, true),
    StructField("site_domain", StringType, true),
    StructField("site_category", StringType, true),
    StructField("app_id", StringType, true),
    StructField("app_domain", StringType, true),
    StructField("app_category", StringType, true),
    StructField("device_id", StringType, true),
    StructField("device_ip", StringType, true),
    StructField("device_model", StringType, true),
    StructField("device_type", IntegerType, true),
    StructField("device_conn_type", IntegerType, true),
    StructField("C14", IntegerType, true),
    StructField("C15", IntegerType, true),
    StructField("C16", IntegerType, true),
    StructField("C17", IntegerType, true),
    StructField("C18", IntegerType, true),
    StructField("C19", IntegerType, true),
    StructField("C20", IntegerType, true),
    StructField("C21", IntegerType, true)
  ))

  case class ClickThroughRatePredictionParams(
    trainInput: String = null,
    testInput: String = null,
    resultOutput: String = null
  )

  /**
   * Try Kaggle's Click-Through Rate Prediction with Logistic Regression Classification
   * Run with
   * {{
   * $SPARK_HOME/bin/spark-submit \
   *   --class org.apache.spark.examples.kaggle.ClickThroughRatePredictionWitLogisticRegression \
   *   /path/to/click-through-rate-prediction-assembly-1.1.jar \
   *   --train=/path/to/train \
   *   --test=/path/to/test \
   *   --result=/path/to/result.csv
   * }}
   * SEE ALSO: https://www.kaggle.com/c/avazu-ctr-prediction
   */
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName)
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val defaultParam = new ClickThroughRatePredictionParams()
    val parser = new OptionParser[ClickThroughRatePredictionParams](this.getClass.getSimpleName) {
      head(s"${this.getClass.getSimpleName}: Try a Kaggle competition.")
      opt[String]("train")
        .text("train input")
        .action((x, c) => c.copy(trainInput = x))
        .required()
      opt[String]("test")
        .text("test input")
        .action((x, c) => c.copy(testInput = x))
        .required()
      opt[String]("result")
        .text("result output path")
        .action((x, c) => c.copy(resultOutput = x))
        .required()
    }
    parser.parse(args, defaultParam).map { params =>
      run(sc, sqlContext, params.trainInput, params.testInput, params.resultOutput)
    } getOrElse {
      sys.exit(1)
    }
    sc.stop()
  }

  def run(sc: SparkContext, sqlContext: SQLContext,
      trainPath: String, testPath: String, resultPath: String): Unit = {
    import sqlContext.implicits._

    // Sets the target variables
    val targetVariables = Array(
      "banner_pos", "site_id", "site_domain", "site_category",
      "app_domain", "app_category", "device_model", "device_type", "device_conn_type",
      "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"
    )

    // Loads training data and testing data from CSV files
    val train = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .schema(trainSchema)
      .load(trainPath).cache()
    val test = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .schema(testSchema)
      .load(testPath).cache()

    // Union data for one-hot encoding
    // To extract features throughly, union the training and test data.
    // Since the test data includes values which doesn't exists in the training data.
    val train4union = train.select(targetVariables.map(col): _*)
    val test4union = test.select(targetVariables.map(col): _*)
    val union = train4union.unionAll(test4union).cache()

    // Extracts features with one-hot encoding
    def getIndexedColumn(clm: String): String = s"${clm}_indexed"
    def getColumnVec(clm: String): String = s"${clm}_vec"
    val feStages = ArrayBuffer.empty[PipelineStage]
    targetVariables.foreach { clm =>
      val stringIndexer = new StringIndexer()
        .setInputCol(clm)
        .setOutputCol(getIndexedColumn(clm))
        .setHandleInvalid("error")
      val oneHotEncoder = new OneHotEncoder()
        .setInputCol(getIndexedColumn(clm))
        .setOutputCol(getColumnVec(clm))
      Array(stringIndexer, oneHotEncoder)
      feStages.append(stringIndexer)
      feStages.append(oneHotEncoder)
    }
    val va = new VectorAssembler()
      .setInputCols(targetVariables.map(getColumnVec))
      .setOutputCol("features")
    feStages.append(va)
    val fePipeline = new Pipeline().setStages(feStages.toArray)
    val feModel = fePipeline.fit(union)
    val trainDF = feModel.transform(train).select('click, 'features).cache()
    val testDF = feModel.transform(test).select('id, 'features).cache()
    union.unpersist()
    train.unpersist()
    test.unpersist()

    // Trains a model with CrossValidator
    val si4click = new StringIndexer()
      .setInputCol("click")
      .setOutputCol("label")
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(si4click, lr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.threshold, Array(0.22))
      .addGrid(lr.elasticNetParam, Array(0.0))
      .addGrid(lr.regParam, Array(0.001))
      .addGrid(lr.maxIter, Array(100))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
    val cvModel = cv.fit(trainDF)

    // Shows the best parameters
    cvModel.bestModel.parent match {
      case pipeline: Pipeline =>
        pipeline.getStages.zipWithIndex.foreach { case (stage, index) =>
          println(s"Stage[${index + 1}]: ${stage.getClass.getSimpleName}")
          println(stage.extractParamMap())
        }
    }

    // Predicts with the trained best model
    val resultDF = cvModel.transform(testDF).select('id, 'probability).map {
      case Row(id: String, probability: Vector) => (id, probability(1))
    }.toDF("id", "click")

    // Save the result
    resultDF.repartition(1).write.mode(SaveMode.Overwrite)
      .format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true")
      .save(resultPath)
  }
}

// scalastyle:on println
