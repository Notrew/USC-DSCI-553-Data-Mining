#!/usr/bin/env python
# coding: utf-8
import sys
from pyspark import SparkContext
import json
import time

# input format: task3.py <review_filepath> <business_filepath> <output_filepath_question_a> <output_filepath_question_b>
review_path = sys.argv[1]
business_path = sys.argv[2]
output_path_a = sys.argv[3]
output_path_b = sys.argv[4]
# review_path = "./data/test_review.json"
# business_path = "./data/business.json"
# output_path_a = "./data/task3_a.txt"
# output_path_b = "./data/task3_b.json"
sc = SparkContext("local[*]",appName="task3").getOrCreate()
review = sc.textFile(review_path).map(lambda x: json.loads(x))
business = sc.textFile(business_path).map(lambda x: json.loads(x))
output = {}

# Question:A
# select business_id and stars form review
# select business_id and city form business
star = review.map(lambda x: (x["business_id"],x["stars"]))
city = business.map(lambda x: (x["business_id"],x["city"]))

# join city and stars with same business_id
joined = city.leftOuterJoin(star)

# calculate average stars of each city
# (('1SWheh84yJXfytovILXOAQ', ('Phoenix', None))
avg_star = joined.map(lambda x: (x[1][0],x[1][1])).filter(lambda x: x[1] is not None).groupByKey()\
                .map(lambda x: (x[0],sum(x[1]),len(x[1]))).map(lambda x: (x[0],float(x[1]/x[2])))\
                .sortBy(lambda x:[-x[1],x[0]]).collect()

# save text file
print(avg_star)
with open(output_path_a,"w",encoding="utf-8") as f:
    f.write("city,stars\n")
    for i in avg_star:
        f.write(str(i[0])+","+str(i[1])+"\n")

# Question:B
#  loading time + time to create joined table
s_time = time.time()
review = sc.textFile(review_path).map(lambda x: json.loads(x))
business = sc.textFile(business_path).map(lambda x: json.loads(x))
star = review.map(lambda x: (x["business_id"],x["stars"]))
city = business.map(lambda x: (x["business_id"],x["city"]))
joined = city.leftOuterJoin(star)
e_time = time.time()
l_time = e_time-s_time

# M1, using python sorting
s_time_1 = time.time()
avg_star_1 = joined.map(lambda x: (x[1][0],x[1][1])).filter(lambda x: x[1] is not None).groupByKey()\
                .map(lambda x: (x[0],sum(x[1]),len(x[1]))).map(lambda x: (x[0],float(x[1]/x[2]))).collect()
avg_star_1 = sorted(avg_star_1,key=lambda x: [-x[1],x[0]])
cnt = 0
for i in range(len(avg_star_1)):
    if cnt < 10:
        print(avg_star_1[i][0])
        cnt += 1
    else:
        break
e_time_1 = time.time()
m1 = l_time+e_time_1-s_time_1

# M2, using spark sorting
s_time_2 = time.time()
avg_star_2 = joined.map(lambda x: (x[1][0],x[1][1])).filter(lambda x: x[1] is not None).groupByKey()\
                .map(lambda x: (x[0],sum(x[1]),len(x[1]))).map(lambda x: (x[0],float(x[1]/x[2])))\
                .sortBy(lambda x:[-x[1],x[0]]).take(10)
cnt = 0
for i in range(len(avg_star_2)):
    if cnt < 10:
        print(avg_star_2[i][0])
        cnt += 1
    else:
        break
e_time_2 = time.time()
m2 = l_time+e_time_2-s_time_2

output["m1"] = m1
output["m2"] = m2
output["reason"] = "Because when sort in spark is done within partitions, so it can work on several partitions at one \
time, but python can only work on one memory, so spark is faster"
print(output)
with open(output_path_b,"w") as f:
    json.dump(output,f) 

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task3.py
# "../resource/asnlib/publicdata/test_review.json" "../resource/asnlib/publicdata/business.json"
# "./task3_a.txt" "./task3_b.json"

