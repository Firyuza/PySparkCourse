from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator


spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet('/home/firiuza/PycharmProjects/PySparkCourse/hmp.parquet')

vectorAssembler = VectorAssembler(inputCols=['x', 'y', 'z'],
                                  outputCol='features')

kmeans = KMeans(k=2, seed=1)

pipeline = Pipeline(stages=[vectorAssembler, kmeans])

model = pipeline.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
