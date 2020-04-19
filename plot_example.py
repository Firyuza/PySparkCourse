import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet('/home/firiuza/PycharmProjects/PySparkCourse/hmp.parquet')

df_x = df.rdd.map(lambda values: values.x).collect()

plt.boxplot(df_x)
plt.show()