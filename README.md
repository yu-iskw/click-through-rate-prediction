# Try Kaggle's Click Through Rate Prediction with Spark Pipeline API

The purpose of this Spark Application is to test Spark Pipeline API with real data.

## The Kaggle Contest

> Predict whether a mobile ad will be clicked
> In online advertising, click-through rate (CTR) is a very important metric for evaluating ad performance. As a result, click prediction systems are essential and widely used for sponsored search and real-time bidding.

https://www.kaggle.com/c/avazu-ctr-prediction


## Approach

1. Extracts features of categorical features with `OneHotEncoder` with `StringIndexer`
2. Train a model with `LogisticRegression` with `CrossValidator`

We merged the training data with the test data in the extracting features phase.
Since, the test data includes values which doesn't exists in the training data.
Therefore, we needed to avoid errors about missing values of each variables, when extracting features of the test data.

## Result

I got the score: `0.3998684` with the following parameter set.

- Logistic Regression
    - `threshold`: 0.22
    - `elasticNetParam`: 0.0
    - `regParam`: 0.01
    - `maxIter`: 100
