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

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}

object ClickThroughRatePredictionWithLogisticRegression {

  val categoricalColumns = Array(
    "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_domain", "app_category",
    "device_model", "device_type", "device_conn_type",
    "C1", "C14", "C15", "C16", "C17",
    "C18", "C19", "C20", "C21"
  )

  case class ClickThroughRatePredictionParams(
    trainInput: String = null,
    testInput: String = null,
    resultOutput: String = null
  )

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
        .text("test input")
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
    trainPath: String, testPath: String, savedPath: String): Unit = {
    import sqlContext.implicits._

    // Loads training data and testing data
    val train = sqlContext.read.format("com.databricks.spark.csv").
      option("header", "true").option("inferSchema", "true").
      load(trainPath).cache()
    val test = sqlContext.read.format("com.databricks.spark.csv").
      option("header", "true").option("inferSchema", "true").
      load(testPath).cache()

    def getIndexedColumn(clm: String): String = s"${clm}_indexed"
    def getColumnVec(clm: String): String = s"${clm}_vec"

    // Formats data
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
    val trainDF = model1.transform(train).select($"label", $"features").cache()
    val testDF = model1.transform(test).select($"id", $"features").cache()

    // Trains a model
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(lr))
    val paramGrid = new ParamGridBuilder().
      addGrid(lr.threshold, Array(0.25, 0.5, 0.75)).
      addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).
      addGrid(lr.fitIntercept, Array(false, true)).
      addGrid(lr.maxIter, Array(50, 100, 150)).
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
    val result = cvModel.transform(testDF)
    result.write.mode(SaveMode.Overwrite).parquet(savedPath)
  }
}
