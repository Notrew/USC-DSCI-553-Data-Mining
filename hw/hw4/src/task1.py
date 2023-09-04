#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from graphframes import GraphFrame
import time
from itertools import combinations


# community detection based on GraphFrames
threshold = int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3]
# threshold = 2
# input_path = "../data/input/ub_sample_data.csv"
# output_path = "../data/output/task1.txt"

s_time = time.time()
sc = SparkContext("local[*]",appName="task1").getOrCreate()
sc.setLogLevel("ERROR")

# read data and exclude the first line of name
# data = sc.textFile(input_path)
# head = data.first()
# data = data.filter(lambda x: x!=head)
data = sc.textFile(input_path).filter(lambda x: x!="user_id,business_id") 
uid_bids = data.map(lambda x: x.split(",")).map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y).mapValues(lambda x: set(x))
# filter first time and construct {uid:[bid,bid,...]}
uid_bids_dict = uid_bids.filter(lambda x: len(x[1])>=threshold).collectAsMap()

# construct nodes and edges
# node-->user, edge-->exist if two nodes" common bids >= threshold
nodes = set()
edges = set()

# get user pairs
uids = list(uid_bids_dict.keys())
user_pairs = combinations(uids,2)
# then caculate len(common_bids) and select those cnt>=threshold
valid_pairs = []
for i in user_pairs:
    len_common_bids = len(uid_bids_dict[i[0]].intersection(uid_bids_dict[i[1]]))
    if len_common_bids>=threshold:
        valid_pairs.append((sorted(i),len_common_bids))

users = set()
for pair in valid_pairs:
    users.add(tuple(pair[0])[0])
    users.add(tuple(pair[0])[1])
nodes = [(user,) for user in sorted(users)]

edges = []
for i in valid_pairs:
    pair = tuple(i[0])
    edges.append(pair)
    edges.append(tuple(reversed(pair)))

# transform nodes and edges to dataframe
sqlContext = SQLContext(sc)
df_nodes = sqlContext.createDataFrame(nodes,["id"])
df_edges = sqlContext.createDataFrame(edges,["src", "dst"])

# consrtuct graph
graph = GraphFrame(df_nodes,df_edges)
# print(graph)

graph = graph.labelPropagation(maxIter=5)
res = graph.rdd.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y).map(lambda x: sorted(x[1]))
# sort by size, then first uid lexicographical
res = res.sortBy(lambda x: (len(x),x)).collect()

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to txt, format: uid1, uid2, uid3,...
with open(output_path,"w") as f:
    for i in res:
        output = ""
        for node in i:
            output = output+"'"+str(node)+"', "
        f.write(output[:-2]+"\n")

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  

# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit 
# --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 --executor-memory 4G --driver-memory 4G 
# task1.py 2 "../resource/asnlib/publicdata/ub_sample_data.csv" "./task1.txt"

