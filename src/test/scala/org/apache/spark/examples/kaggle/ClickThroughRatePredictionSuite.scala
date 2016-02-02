package org.apache.spark.examples.kaggle

import org.apache.spark.SparkFunSuite
import org.apache.spark.util.MLlibTestSparkContext

class ClickThroughRatePredictionSuite extends SparkFunSuite with MLlibTestSparkContext {

  val trainPath = this.getClass.getResource("/train.part-10000").getPath
  val testPath = this.getClass.getResource("/test.part-10000").getPath

  test("run") {
    //    Logger.getLogger("org").setLevel(Level.OFF)
    //    Logger.getLogger("akka").setLevel(Level.OFF)

    ClickThroughRatePrediction.run(sc, sqlContext, trainPath, testPath)
  }
}
