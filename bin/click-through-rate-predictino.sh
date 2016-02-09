#!/bin/bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

THIS_PROJECT_HOME="$( cd $(dirname $( dirname $0)) && pwd )"

PACKAGED_JAR=$(find ${THIS_PROJECT_HOME}/target -name "click-through-rate-prediction-assembly*.jar")

${SPARK_HOME}/bin/spark-submit \
  --class "org.apache.spark.examples.kaggle.ClickThroughRatePrediction" \
  "$PACKAGED_JAR" \
  --train "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/train" \
  --test "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/test"
  --result "s3n://s3-yu-ishikawa/test-data/click-through-rate-prediction/result"
