#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import json
import time

# task2, input format: task2.py <review_filepath> <output_filepath> <n_partition>
review_path = sys.argv[1]
output_path = sys.argv[2]
n_partition = int(sys.argv[3])
# review_path = "./data/test_review.json"
# output_path = "./data/task2.json"
# n_partition = 4
sc = SparkContext("local[*]",appName="task2").getOrCreate()
review = sc.textFile(review_path).map(lambda x: json.loads(x))
output = {}

# default
default = {}
s_time = time.time()
review.map(lambda x: [x["business_id"],1]).reduceByKey(lambda x,y: x+y).sortBy(lambda x:[-x[1],x[0]]).take(10)
e_time = time.time()

default["n_partition"] = review.getNumPartitions()
default["n_items"] = review.glom().map(lambda x: len(x)).collect()
# default["n_items"] = review.mapPartitions(lambda x: [len(list(x))]).collect()
default["exe_time"] = e_time-s_time

# customized
customized = {}
s_time = time.time()
review_new = review.map(lambda x: (x['business_id'], 1)).partitionBy(n_partition, lambda x: ord(x[0]))
review_new.reduceByKey(lambda x,y: x+y).sortBy(lambda x:[-x[1],x[0]]).take(10)
e_time = time.time()

customized["n_partition"] = review_new.getNumPartitions()
customized["n_items"] = review_new.glom().map(lambda x: len(x)).collect()
customized["exe_time"] = e_time-s_time

# output
output["default"] = default
output["customized"] = customized

print(output)
with open(output_path,"w") as f:
    json.dump(output,f) 

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2.py
# "../resource/asnlib/publicdata/test_review.json" "./task2.json" 3

