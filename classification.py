from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet('/home/firiuza/PycharmProjects/PySparkCourse/hmp.parquet')

splits = df.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

indexer = StringIndexer(inputCol='class', outputCol='label').fit(df)
vectorAssembler = VectorAssembler(inputCols=['x', 'y', 'z'],
                                  outputCol='features')
normalizer = Normalizer(inputCol='features', outputCol='features_norm', p=1.0)


def log_reg():
    log_reg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer, log_reg])

    model = pipeline.fit(df_train)

    predictions = model.transform(df_train)

    eval = MulticlassClassificationEvaluator().setMetricName('accuracy') \
        .setLabelCol('label')

    print('train: %f' % eval.evaluate(predictions))

    model = pipeline.fit(df_test)
    predictions = model.transform(df_test)

    print('test: %f' % eval.evaluate(predictions))

    return

def random_forest():
    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features_norm", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=indexer.labels)

    pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(df_train)

    predictions = model.transform(df_test)

    predictions.select("predictedLabel", "label", "features_norm").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    return

random_forest()
