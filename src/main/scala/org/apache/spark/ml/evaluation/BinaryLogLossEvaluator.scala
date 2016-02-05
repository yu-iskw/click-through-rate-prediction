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
package org.apache.spark.ml.evaluation

import org.apache.commons.math3.util.FastMath

import org.apache.spark.annotation.Since
import org.apache.spark.ml.param.shared.{HasLabelCol, HasProbabilityCol}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row}

/**
 * Evaluator for probability prediction, using logarithmic loss,
 * which expects two input columns: probability and label.
 */
class BinaryLogLossEvaluator(override val uid: String)
  extends Evaluator with HasProbabilityCol with HasLabelCol {

  def this() = this(Identifiable.randomUID("binLogLossEval"))

  /**
   * param for epsilon in evaluation
   * Log loss is undefined for p=0 or p=1, so probabilities are
   * clipped to max(epsilon, min(1 - espsilon, p)).
   *
   * Default: 10e-15
   * @group expertParam
   */
  @Since("2.0.0")
  val epsilon: Param[Double] = new Param(this, "epsilon", "epsilon", (x: Double) => x > 0.0)
  setDefault(epsilon -> 10e-15)

  /** @group setParam */
  @Since("2.0.0")
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  /** @group setParam */
  @Since("2.0.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group expertSetParam */
  @Since("2.0.0")
  def setEpsilon(value: Double): this.type = set(epsilon, value)

  /** @group expertGetParam */
  @Since("2.0.0")
  def getEpsion: Double = $(epsilon)

  @Since("2.0.0")
  override def isLargerBetter: Boolean = false

  @Since("2.0.0")
  override def copy(extra: ParamMap): BinaryLogLossEvaluator = defaultCopy(extra)

  @Since("2.0.0")
  override def evaluate(dataset: DataFrame): Double = {
    val schema = dataset.schema
    SchemaUtils.checkColumnType(schema, $(probabilityCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)

    // TODO: When dataset metadata has been implemented, check rawPredictionCol vector length = 2.
    val epsilon = getEpsion
    val minusLogLoss = dataset.select($(probabilityCol), $(labelCol))
      .map { case Row(probabilities: Vector, label: Double) =>
        val probability = Math.max(epsilon, Math.min(1 - epsilon, probabilities(1)))
        label * FastMath.log(probability) + (1 - label) * FastMath.log(1 - probability)
      }.mean()
    -1.0 * minusLogLoss
  }
}
