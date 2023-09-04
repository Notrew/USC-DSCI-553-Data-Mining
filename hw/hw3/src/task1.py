#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import time
from itertools import combinations
import random
import csv

# implement the LSH with Jaccard similarity on yelp_train.csv
# task1.py <input_file_path> <output_file_path>
input_path = sys.argv[1]
output_path = sys.argv[2]
# input_path = "../data/input/yelp_train.csv"
# output_path = "../data/output/task1.csv"

# design hash function to shuffle uids
def shuffleIndex(shuffle_times,indexToShuffle):
    index_shuffled_n_times = []
    # f(x) = ((ax + b) % p) % m
    a = random.sample(range(1,round(time.time()/10000)),shuffle_times)
    b = random.sample(range(1,round(time.time()/10000)),shuffle_times)
    p = 9965
    m = len(indexToShuffle)+1 #number of bins
    for i_th_shuffle in range(shuffle_times):
        new_index_l = []
        for index in range(len(indexToShuffle)):
            new_inex  = ((a[i_th_shuffle]*index + b[i_th_shuffle])%p)%m
            new_index_l.append(new_inex)
        index_shuffled_n_times.append(new_index_l)
    return index_shuffled_n_times

# select the minimum uid index of one hashed results for one bid
# and then to build signature matrix
def minHash(valid_item_l,org_index_l,index_shuffled_n_times):
    # get original index of uid
    org_index = [org_index_l[i] for i in valid_item_l]
    sig = []
    for i_th_shuffle in range(len(index_shuffled_n_times)):
        new_index = []
        for index in org_index:
            new_index.append(index_shuffled_n_times[i_th_shuffle][index])
        sig.append(min(new_index))
    return sig

def genCandPairs(b_num,r_num,list_of_uid_list):
    res = set()
    # divide sig_matrix into b bands
    for band_i in range(b_num):
        start_row = int(band_i*2)
        in_one_band = {}  # {[uid,uid,...]:[bid,bid,...]}
        for i_th_bid_index in range(len(list_of_uid_list)):
            its_uid_indexes = list_of_uid_list[i_th_bid_index]
            portion_uid_indexes = tuple(its_uid_indexes[start_row:start_row+r_num]) #list is unhashable
            # for each band, hash bids with same portion of uids in to one bucket
            if portion_uid_indexes in in_one_band:
                in_one_band[portion_uid_indexes].append(i_th_bid_index)
            else:
                in_one_band[portion_uid_indexes] = [i_th_bid_index]
        # candidate paris are those that hash to same buckets more than one bands
        # in in_one_band dict, bids in one list are possible sigletons of candidate pairs
        for bid_l in list(in_one_band.values()):
            if len(bid_l) >= 2:
                pairs = combinations(bid_l,2)
                # sort and save in res set
                for pair in pairs:
                    res.add(tuple(sorted(pair)))
        # do combinations after going through all bands and removing duplicates
        # try it latter
    return res

def cal_sim_and_filter(candidate_pairs,threshold):
    res = []
    for pair in candidate_pairs:
        # # get corresponding uid list according to bid
        # uids_of_bid_1 = bid_uids_info_dict[bid_1]
        # uids_of_bid_2 = bid_uids_info_dict[bid_2]
        # get corresponding uid list according to bid index
        uids_of_bid_1 = set(bid_index_uids_info_dict[pair[0]])
        uids_of_bid_2 = set(bid_index_uids_info_dict[pair[1]])
        # calculate jaccard similarity
        j_sim = len(uids_of_bid_1.intersection(uids_of_bid_2))/len(uids_of_bid_1.union(uids_of_bid_2))
        # filter those j_sim >= threshold
        if j_sim >= threshold:
            # get corresponding bid
            bid_1 = bid_index[pair[0]]
            bid_2 = bid_index[pair[1]]
            res.append(sorted([bid_1,bid_2])+[j_sim])
    return sorted(res)

s_time = time.time()
sc = SparkContext("local[*]",appName="task1").getOrCreate()

data = sc.textFile(input_path).filter(lambda x: x!="user_id,business_id,stars") #exclude the first line of name
# split one row into (uid,bid)
uid_bid = data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1]))
# combine uid of the same bid into a list==>(bid,[uid,uid,...])
bid_uids = uid_bid.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y)
# remove the dupplicates
bid_uids = bid_uids.mapValues(lambda x: [*set(x)])

# # get bid list and bid index, used to get bid from index
bids = uid_bid.map(lambda x: x[1]).distinct().collect()
bid_index = {}
for i in range(len(bids)):
    bid_index[i] = bids[i]

# get uid list and uid index, shuffle latter
uids = uid_bid.map(lambda x: x[0]).distinct().collect()
uid_index = {}
for i in range(len(uids)):
    uid_index[uids[i]] = i

# # get bid and corresponding uid list into dict, used to calculate simmilarity according to bid
# bid_uids_info_list = bid_uids.collect()
# bid_uids_info_dict = {}
# for i in range(len(bid_uids_info_list)):
#     bid_uids_info_dict[bid_uids_info_list[i][0]] = bid_uids_info_list[i][1]

# get bid_index and corresponding uid list into dict, used to calculate simmilarity according to bid_index
bid_uids_info_list = bid_uids.collect()
bid_index_uids_info_dict = {}
for i in range(len(bid_uids_info_list)):
    bid_index_uids_info_dict[i] = bid_uids_info_list[i][1]

# hash index
shuffle_times = 50
index_shuffled_50_times = shuffleIndex(shuffle_times,uids)

# minhash, generage signature matrix (50 x len_of_bid)
sig_matrix = bid_uids.map(lambda x: (x[0],minHash(x[1],uid_index,index_shuffled_50_times)))

# LSH
# combine uid lists into one list
sig_matrix_comb_uid = sig_matrix.map(lambda x: (0,x[1])).groupByKey().map(lambda x: list(x[1]))
# generage candidate pairs, (bid_index,bid_index)
# b bands and r rows, b*r=n(number of hash functions)
b_num = 25
r_num = 2
# genCandPairs(b_num,r_num,sig_matrix_comb_uid.collect()[0])
candidate_pairs = sig_matrix_comb_uid.map(lambda x: genCandPairs(b_num,r_num,x))

# for i in range(len(bid_index)):
#     while not bid_index_uids_info_dict[i]==bid_uids_info_dict[bid_index[i]]:
#         print("ha")

# filter candidate pairs whose Jaccard similarity is >= 0.5
threshold = 0.5
pair_sim = candidate_pairs.map(lambda x: cal_sim_and_filter(x,threshold)).collect()[0]

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, header: business_id_1, business_id_2, similarity
with open(output_path,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["business_id_1","business_id_2","similarity"])
    for i in pair_sim:
        # print(i)
        writer.writerow(i)

# calculate precision and recall score based on ground truth file “pure_jaccard_similarity.csv”
# precision >= 0.99 and recall >= 0.9
# Precision = true positives / (true positives + false positives) 

# import pandas as pd
# from sklearn.metrics import precision_score, recall_score
# task1 = pd.read_csv(output_path)
# truth = pd.read_csv("../data/input/pure_jaccard_similarity.csv")
# task1["comb"] = task1.apply(lambda row: row["business_id_1"]+row["business_id_2"]+str(row["similarity"]),axis = 1)
# truth["comb"] = truth.apply(lambda row: row["business_id_1"]+row[" business_id_2"]+str(row[" similarity"]),axis = 1)

# print(precision_score(truth["comb"],task1["comb"],average='macro'))
# print(precision_score(truth["comb"],task1["comb"],average='micro'))
# print(precision_score(truth["comb"],task1["comb"],average="weighted"))

# # Recall = true positives / (true positives + false negatives)
# print(recall_score(truth["comb"],task1["comb"],average='macro'))
# print(recall_score(truth["comb"],task1["comb"],average='micro'))
# print(recall_score(truth["comb"],task1["comb"],average="weighted"))

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task1.py
# "../resource/asnlib/publicdata/yelp_train.csv"
# "./task1.csv"

