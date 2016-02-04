package org.apache.spark.examples.kaggle

import org.apache.spark.SparkFunSuite
import org.apache.spark.util.MLlibTestSparkContext

class ClickThroughRatePredictionWithRandomForestClassificationSuite extends SparkFunSuite
    with MLlibTestSparkContext {

  val trainPath = this.getClass.getResource("/train.part-100000").getPath
  val testPath = this.getClass.getResource("/test.part-10000").getPath
  val savedPath = "./tmp/result/"

  test("run") {
    //    Logger.getLogger("org").setLevel(Level.OFF)
    //    Logger.getLogger("akka").setLevel(Level.OFF)

    ClickThroughRatePredictionWithRandomForestClassification
      .run(sc, sqlContext, trainPath, testPath, savedPath)
  }
}
