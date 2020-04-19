import numpy as np
import pyspark
import os
import time
import random

from collections import namedtuple
from pyspark import SparkContext, SparkConf, AccumulatorParam
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import lit
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField('x', IntegerType(), True),
    StructField('y', IntegerType(), True),
    StructField('z', IntegerType(), True)
])

file_list = os.listdir('/home/firiuza/MachineLearning/HMP_Dataset')
file_list_filtered = [s for s in file_list if '_' in s]

df = None

for category in file_list_filtered:
    data_files = os.listdir('/home/firiuza/MachineLearning/HMP_Dataset/%s' % category)

    for data_file in data_files:
        print(data_file)

        temp_df = spark.read.option('header', 'false')\
            .option('delimiter', ' ')\
            .csv('/home/firiuza/MachineLearning/HMP_Dataset/%s/%s' % (category, data_file),
                 schema=schema)

        temp_df = temp_df.withColumn('class', lit(category))
        temp_df = temp_df.withColumn('source', lit(data_file))

        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)

# Save data in parquet format
# df.write.repartition(1).parquet('hmp.parquet')

indexer = StringIndexer(inputCol='class', outputCol='classIndex')
# indexed = indexer.fit(df).transform(df)

encoder_oh = OneHotEncoder(inputCol='classIndex', outputCol='categoryVec')
# encoded = encoder_oh.fit(indexed).transform(indexed)

vectorAssembler = VectorAssembler(inputCols=['x', 'y', 'z'],
                                  outputCol='features')
# features_vectorized = vectorAssembler.transform(encoded)

normalizer = Normalizer(inputCol='features', outputCol='features_norm', p=1.0)
# normalized_data = normalizer.transform(features_vectorized)


pipeline = Pipeline(stages=[indexer, encoder_oh, vectorAssembler, normalizer])

model = pipeline.fit(df)

prediction = model.transform(df)

df_train = prediction.drop('x').drop('y').drop('z')\
    .drop('class').drop('source').drop('features')

df_train.show()





