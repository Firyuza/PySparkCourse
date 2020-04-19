import numpy as np
import pyspark
import os
import time
import random

from collections import namedtuple
from pyspark import SparkContext, SparkConf, AccumulatorParam
from datetime import datetime, timedelta

class MaxAccumulatorParam(AccumulatorParam):
    def zero(self, initial_value):
        return initial_value

    def addInPlace(self, accumulator, delta):
        return max(accumulator, delta)


def parse_record(s):
    fields = s.split(",")

    return Record(fields[0], *map(float, fields[1:6]), int(fields[6]))


def volume_accum():
    # print(parsed_data.map(lambda x: x.Date).min())

    # print(parsed_data.map(lambda x: x.Date).max())

    # print(parsed_data.map(lambda x: x.Volume).min())

    with_month_data = parsed_data.map(lambda x: (x.Date[:7], x))
    print(with_month_data.take(1))

    by_month_data = with_month_data.mapValues(lambda x: x.Volume)
    by_month_data = by_month_data.reduceByKey(lambda accum, n: accum + n)
    print(by_month_data.top(1, lambda x: x[1]))

    result_data = by_month_data.map(lambda t: ','.join(map(str, t)))
    print(result_data.take(1))

    result_data.repartition(1).saveAsTextFile('out')

    return

def get_next_date(s):
    fmt = '%Y-%m-%d'

    return (datetime.strptime(s, fmt) + timedelta(days=1)).strftime(fmt)

def daily_return_ratio(parsed_data):
    date_and_close_price = parsed_data.map(lambda r: (r.Date, r.Close))
    print(date_and_close_price.take(2))

    date_and_prev_close_price = parsed_data.map(lambda r: (get_next_date(r.Date), r.Close))
    print(date_and_prev_close_price.take(2))

    joined = date_and_close_price.join(date_and_prev_close_price)
    print(joined.take(1))

    returns = joined.mapValues(lambda p: (p[0] / p[1] - 1.) * 100.)
    print(returns.take(1))

    return

def accumulator_func(sc, parsed_data):
    time_persist = sc.accumulator(0.0, MaxAccumulatorParam())

    def persist_to_external_storage(iterable):
        for record in iterable:
            before = time.time()
            time.sleep(random.random() / 1000.0)
            after = time.time()

            time_persist.add(after - before)


    parsed_data.foreachPartition(persist_to_external_storage)

    print(time_persist.value)

    return

# appName = "demo1"
# master = "local1"
# conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext.getOrCreate()
# sc = SparkContext('local')


Record = namedtuple("Record", ['Date', 'Open', 'High' ,
                               'Low' , 'Close', 'Adj_Close', 'Volume'])

parsed_data = sc.textFile(name="./IXIC.csv")\
    .map(parse_record)\
    .cache()

# accumulator_func(sc, parsed_data)
daily_return_ratio(parsed_data)