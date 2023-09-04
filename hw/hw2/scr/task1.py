#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import time
from itertools import combinations
import math

# task1.py <case number> <support> <input_file_path> <output_file_path>
# case_num = int(sys.argv[1])
# s = int(sys.argv[2])
# input_path = sys.argv[3]
# output_path = sys.argv[4]
case_num = 2
s = 9
input_path = "../data/small2.csv"
output_path = "../data/output/task1.txt"

s_time = time.time()
sc = SparkContext("local[*]",appName="task1").getOrCreate()
data = sc.textFile(input_path).filter(lambda x: x!="user_id,business_id") #exclude the first line of name

if case_num == 1:
    # merge bysiness id
    data = data.map(lambda x: (x.split(",")[0],[x.split(",")[1]])).reduceByKey(lambda x,y: x+y)
else:
    data = data.map(lambda x: (x.split(",")[1],[x.split(",")[0]])).reduceByKey(lambda x,y: x+y)
# remove the dupplicates
data = data.mapValues(lambda x: [*set(x)])
baskets_l = data.map(lambda x: x[1])

# get frequent singletons
# baskets_l.glom().flatMap(lambda x: freq_single(x,4)).distinct().collect()
def get_freq_single(baskets,threshold):
    counts = {}
    freq_singles = []
    # count frequency
    for basket in baskets:
        for singletons in basket:
            # if singletons in counts:
            #     counts[singletons] += 1
            # else:
            #     counts[singletons] = 1
            counts[singletons] = counts.get(singletons,0)+1
    # select candidate that counts over threshold
    for candicate in counts.keys():
        if counts[candicate] >= threshold:
            freq_singles.append(candicate)
    return freq_singles

# get frequent 1 to k pairs
# baskets_l.glom().flatMap(lambda x: get_all_freq_pairs(x,4,3)).distinct().collect()
def get_all_freq_pairs(baskets,threshold,max_len):
    freq_singles = get_freq_single(baskets,threshold)
    if max_len == 1:
        return freq_singles
    # max_len > 1
    prev_freq = freq_singles
    all_freq_pairs = [(i,) for i in freq_singles]
    pair_size = 2
    while pair_size <= max_len:
        counts = {}
        freq_k_pair = []
        if pair_size == 2:
            accu_prev_freq = freq_singles
        else:
            accu_prev_freq = set()
            for pair in prev_freq:
                for item in pair:
                    accu_prev_freq.add(item)
        # construct pari of size k by previous frequent items/pairs
        # k_pair = combinations(accu_prev_freq,pair_size)
        # # count frequency
        # for pair in k_pair:
        #     for basket in baskets:
        #         # check if pair in single basket
        #         if all(x in basket for x in pair):
        #             counts[pair] = counts.get(pair,0)+1
        # or
        for basket in baskets:
            basket = sorted(set(basket).intersection(set(accu_prev_freq)))
            k_pair = combinations(basket,pair_size)
            for pair in k_pair:
                # pair = tuple(pair)
                counts[pair] = counts.get(pair,0)+1
        # select candidate pair that counts over threshold
        for candicate in counts.keys():
            if counts[candicate] >= threshold:
                freq_k_pair.append(candicate)
        # remove duplicates
        freq_k_pair = [tuple(sorted(i)) for i in freq_k_pair]
        freq_k_pair = sorted(list(set(freq_k_pair)))
        all_freq_pairs += freq_k_pair
        # update next pari size
        pair_size += 1
        prev_freq = freq_k_pair
    return all_freq_pairs

# implment son alg: find n(item)>new_threshold
def son_alg(baskets,baskets_size,support):
    max_len = max(len(i) for i in baskets)
    new_threshold = math.ceil(len(baskets)/baskets_size*support)
    all_freq_pairs = get_all_freq_pairs(baskets,new_threshold,max_len)
    return all_freq_pairs

# counts candidate frequent in total
def counts_in_total(baskets,candidates):
    counts = {}
    for item in candidates:
        for basket in baskets:
            if all(x in basket for x in item):
                counts[item] = counts.get(item,0)+1
    res = [(i,counts[i]) for i in counts.keys()]
    return res

# pass 1: get candidate frequent itemset
baskets_size = baskets_l.count()
all_freq = baskets_l.glom().flatMap(lambda x: son_alg(x,baskets_size,s)).distinct().collect()
inter_res = sorted(all_freq, key=lambda x: [len(x), x])

# pass 2: counts candidates and select true frequent
true_freq = baskets_l.glom().flatMap(lambda x: counts_in_total(x,inter_res)).reduceByKey(lambda x,y: x+y).\
    filter(lambda x: x[1]>=s).map(lambda x: x[0]).collect()
final_res = sorted(true_freq, key=lambda x: [len(x), x])

def output_format(res):
    # output = {i+1:[] for i in range(max(len(i) for i in inter_res))}
    output = {}
    for i in res:
        new_item = "('"+"', '".join(list(i))+"'),"
        if len(i) in output:
            output[len(i)] = output[len(i)]+new_item
        else:
            output[len(i)] = new_item
    # max_len = max(len(i) for i in res)
    output_txt = ""
    for i in output.keys():
        # if i != max_len:
        output_txt += output[i][:-1]+"\n\n"
    return output_txt

with open(output_path,"w") as f:
    output_txt = "Candidates:\n"+output_format(inter_res)+"Frequent Itemsets:\n"+output_format(final_res)
    f.write(output_txt[:-2])

e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py
# 1 2
# 4 9
# "../resource/asnlib/publicdata/small1.csv"
# "./task1.txt"

