from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet('/home/firiuza/PycharmProjects/PySparkCourse/hmp.parquet')


df.createOrReplaceTempView('df')
df_energy = spark.sql('''
select sqrt(sum(x*x)+sum(y*Y)+sum(z*z)) as label, class
 from df group by class

''')

df_energy.createOrReplaceTempView('df_energy')

df_join = spark.sql('''
select * from df inner join df_energy on df.class=df_energy.class
''')


vectorAssembler = VectorAssembler(inputCols=['x', 'y', 'z'],
                                  outputCol='features')

normalizer = Normalizer(inputCol='features', outputCol='features_norm', p=1.0)

log_reg = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

pipeline = Pipeline(stages=[vectorAssembler, normalizer, log_reg])

model = pipeline.fit(df_join)

predictions = model.transform(df_join)

print(model.stages[2].summary.r2)