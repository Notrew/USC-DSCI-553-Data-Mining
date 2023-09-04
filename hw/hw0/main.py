# This is a sample Python script.

from pyspark import SparkContext
import os

sc = SparkContext('local[*]', 'wordCount')

data = "hello pyspark hello dsci553"
input_file_path = './text.txt'
input_file = open(input_file_path, 'w')
input_file.write(data)
input_file.close()

textRDD = sc.textFile(input_file_path)
counts = textRDD.flatMap(lambda line:line.split(' '))\
                         .map(lambda word:(word, 1)).reduceByKey((lambda  a,b: a+b)).collect()
for each_word in counts:
    print(each_word)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G main.py
