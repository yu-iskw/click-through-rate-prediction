# Try Kaggle's Click Through Rate Prediction with Spark Pipeline API

The purpose of this Spark Application is to test Spark Pipeline API with real data for [SPARK-13239](https://issues.apache.org/jira/browse/SPARK-13239).
So, we tested ML Pipeline API with Kaggle's click-through rate prediction.


## Build & Run
You can build this Spark application with `sbt clean assembly`.
And you can run it the command.

```
$SPARK_HOME/bin/spark-submit \
  -class org.apache.spark.examples.kaggle.ClickThroughRatePredictionWitLogisticRegression \
  /path/to/click-through-rate-prediction-assembly-1.0.jar \
  --train=/path/to/train \
  --test=/path/to/test \
  --result=/path/to/result.csv
```

- `--train`: the training data you downloaded
- `--test`: the test data you downloaded
- `--result`: result file

You know, Spark ML can't write a single file directly.
However, making the number of partitions of result DataFrame 1, this application aggregates the result as a file.
So you can get the result CSV file from `part-00000` under the path which you set at `--result` option.

## The Kaggle Contest

> Predict whether a mobile ad will be clicked
> In online advertising, click-through rate (CTR) is a very important metric for evaluating ad performance. As a result, click prediction systems are essential and widely used for sponsored search and real-time bidding.

https://www.kaggle.com/c/avazu-ctr-prediction


## Approach

1. Extracts features of categorical features with `OneHotEncoder` with `StringIndexer`
2. Train a model with `LogisticRegression` with `CrossValidator`
    - The `Evaluator` of `CrossValidator` is the default of `BinaryClassificationEvaluator`.

We merged the training data with the test data in the extracting features phase.
Since, the test data includes values which doesn't exists in the training data.
Therefore, we needed to avoid errors about missing values of each variables, when extracting features of the test data.

## Result

I got the score: `0.3998684` with the following parameter set.

- Logistic Regression
    - `featuresCol`: features
    - `fitIntercept`: true
    - `labelCol`: label
    - `maxIter`: 100
    - `predictionCol`: prediction
    - `probabilityCol`: probability
    - `rawPredictionCol`: rawPrediction
    - `regParam`: 0.001
    - `standardization`: true
    - `threshold`: 0.22
    - `tol`: 1.0E-6
    - `weightCol`:

## TODO

We should offer more `Evaluator`s, such as logg-loss.
Since `spark.ml` doesn't offer Loggistic-Loss at Spark 1.6, we might get better score with logg-loss evaluator.
