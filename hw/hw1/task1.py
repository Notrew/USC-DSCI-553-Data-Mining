#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import json

# task1, input format: task1.py <review_filepath> <output_filepath>
review_path = sys.argv[1]
output_path = sys.argv[2]
# review_path = "./data/test_review.json"
# output_path = "./data/task1.json"
sc = SparkContext("local[*]",appName="task1").getOrCreate()
review = sc.textFile(review_path).map(lambda x: json.loads(x))
output = {}

# A. The total number of reviews
output["n_review"] = review.map(lambda x: x["review_id"]).distinct().count()

# B. The number of reviews in 2018
output["n_review_2018"] = review.filter(lambda x: x["date"].startswith("2018")).\
                                map(lambda x: x["review_id"]).distinct().count()


# C. The number of distinct users who wrote reviews
output["n_user"] = review.map(lambda x: x["user_id"]).distinct().count()

# D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
output["top10_user"] = review.map(lambda x: [x["user_id"],1]).reduceByKey(lambda x,y: x+y)\
                        .map(lambda x: list(x)).takeOrdered(10,key=lambda x: [-x[1],x[0]])
# output["top10_user"] = review.map(lambda x: [x["user_id"],1]).reduceByKey(lambda x,y: x+y)\
#                         .map(lambda x: list(x)).takeOrdered(10,key=lambda x: [-x[1],x[0]])

# E. The number of distinct businesses that have been reviewed
output["n_business"] = review.map(lambda x: x["business_id"]).distinct().count()

# F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
output["top10_business"] = review.map(lambda x: [x["business_id"],1]).reduceByKey(lambda x,y: x+y)\
                        .map(lambda x: list(x)).sortBy(lambda x:[-x[1],x[0]]).take(10)
# output["top10_business"] = review.map(lambda x: [x["business_id"],1]).reduceByKey(lambda x,y: x+y)\
#                         .map(lambda x: list(x)).takeOrdered(10,key=lambda x: [-x[1],x[0]])

print(output)
with open(output_path,"w") as f:
    json.dump(output,f) 

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py
# "../resource/asnlib/publicdata/test_review.json"
# "./task1.json"