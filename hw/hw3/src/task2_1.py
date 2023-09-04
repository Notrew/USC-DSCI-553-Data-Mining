#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import time
import math
import csv

# task2_1.py <input_file_path> <test_file_path> <output_file_path>
input_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
# input_path = "../data/input/yelp_train.csv"
# test_path = "../data/input/yelp_val.csv"
# output_path = "../data/output/task2_1.csv"

# Pearson similarity between i and j
sim_cache = {} #sim_cache
def calSim(bid1,bid2):
    # find avg rate for each item--all rated, not co-rated
    avg_rate_1 = hist_bid_avg_rate[bid1]
    avg_rate_2 = hist_bid_avg_rate[bid2]
    user_list_1 = hist_bid_uids_info_dict[bid1]
    user_list_2 = hist_bid_uids_info_dict[bid2]
    # find users both rated i and j
    co_rate_users = set(user_list_1).intersection(set(user_list_2))
    # calculate (rate-avg_rate) for each user on i and j
    nor_rates = []
    for  co_rate_user in co_rate_users:
        nor_rate_1 = hist_bid_uid_tuple_rate[tuple([bid1,co_rate_user])]-avg_rate_1
        nor_rate_2 = hist_bid_uid_tuple_rate[tuple([bid2,co_rate_user])]-avg_rate_2
        nor_rates.append([nor_rate_1,nor_rate_2])
    # calculate Pearson similarity
    nmr = sum([rate[0]*rate[1] for rate in nor_rates])
    dnm = math.sqrt(sum([rate[0]**2 for rate in nor_rates]))*math.sqrt(sum([rate[1]**2 for rate in nor_rates]))
    if dnm != 0:
        sim = nmr/dnm
    else:
        sim = 0
    pair = tuple(sorted([bid1,bid2]))
    if pair not in sim_cache:
        sim_cache[pair] = sim
    return sim

def predict(bid_to_pred,test_bid_uids_info_dict):
    # [[uid,bid,pred_rate],...]
    res = [] 
    # new bid, use all users rated this bid and rate=3.0 to build item profile
    if bid_to_pred not in hist_bids:
        # res = [[uid,bid_to_pred,3.0] for uid in test_bid_uids_info_dict[bid_to_pred]]
        # or use avg_rate of this user to fill latter
        res = [[uid,bid_to_pred,hist_uid_avg_rate[uid]] for uid in test_bid_uids_info_dict[bid_to_pred]]
        return res
    users_to_pred = test_bid_uids_info_dict[bid_to_pred]
    for user in users_to_pred:
        rate_sim = []
        # new user, use rate=3.0 to build item profile
        if user not in hist_uids:
            # res.append([user,bid_to_pred,3.0])
            # or use avg_rate of this bid to fill latter
            res.append([user,bid_to_pred,hist_bid_avg_rate[bid_to_pred]])
            continue
        # bid and user both have historical data
        # if this user only rated bid_to_pred before, use historical data
        if hist_uid_bids_info_dict[user]==[bid_to_pred]:
            res.append([user,bid_to_pred,hist_bid_uid_tuple_rate[(user,bid_to_pred)]])
            continue
        # find possible neighbor/bid
        possible_nbors = set(hist_uid_bids_info_dict[user])-set(bid_to_pred)
        # find co-rated user of bid_to_pred/i and possible_nbor
        for possible_nbor in possible_nbors:
            co_rate_users = set(hist_bid_uids_info_dict[bid_to_pred]).intersection(set(hist_bid_uids_info_dict[possible_nbor]))
            if not co_rate_users:
                continue
            else:
            # calculate sim 
                # if alreadey calculated
                pair = tuple(sorted([bid_to_pred,possible_nbor]))
                if pair in sim_cache:
                    sim = sim_cache[pair]
                else:
                    sim = calSim(bid_to_pred,possible_nbor)
                rate_sim.append([hist_bid_uid_tuple_rate[(possible_nbor,user)],sim])    
        # select top n neighbors
        rate_sim.sort(key=lambda x: x[1],reverse=True)
        n = min(20,len(rate_sim))
        top_nbor_info = rate_sim[:n]
        nmr = sum([info[0]*info[1] for info in top_nbor_info])
        dnm = sum([abs(info[1]) for info in top_nbor_info])
        # predict
        if dnm != 0:
            rate_pred = 0.1*nmr/dnm +0.5*hist_bid_avg_rate[bid_to_pred]+0.4*hist_uid_avg_rate[user]
            rate_pred = min(5.0,max(0.0,rate_pred))
        else:
            rate_pred = (hist_bid_avg_rate[bid_to_pred]+hist_uid_avg_rate[user])/2
        res.append([user,bid_to_pred,rate_pred])
    return res
    
s_time = time.time()
sc = SparkContext("local[*]",appName="task1").getOrCreate()

# Item-based CF recommendation system with Pearson similarity
# step
# Pearson similarity between i and j
    # calculate avg rate for each item--all rated, not co-rated
    # find users both rated i and j
    # calculate (rate-avg_rate) for each user on i and j
# select items with highest similarity as neighbors
# predict

# read train data and train and get basic info
# split one row into (uid,bid,star)
train_data = sc.textFile(input_path)
head = train_data.first()
train_data = train_data.filter(lambda x: x!=head) #exclude the first line of name
hist_uid_bid_rate = train_data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1],float(x[2])))
# combine bid of the same uid into a list and remove the duplicates
# (uid,[bid,bid,...])
hist_uid_bids = hist_uid_bid_rate.map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y).mapValues(lambda x: [*set(x)])
# {uid:['bid,bid,...]}, find neighbors
hist_uid_bids_info_dict = hist_uid_bids.collectAsMap()
hist_uids = list(hist_uid_bids_info_dict.keys())
# (bid,[uid,uid,...])
hist_bid_uids = hist_uid_bid_rate.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y).mapValues(lambda x: [*set(x)])
# {bid:['uid,uid,...]}, find co-rated users
hist_bid_uids_info_dict  = hist_bid_uids.collectAsMap()
hist_bids = list(hist_bid_uids_info_dict.keys())
# {(bid,uid):score}
hist_bid_uid_tuple_rate = hist_uid_bid_rate.map(lambda x: ((x[1],x[0]),x[2])).collectAsMap()
# avg rate for each item--all rated, not co-rated
# {uid:avg_star}
hist_uid_avg_rate = hist_uid_bid_rate.map(lambda x: (x[0],x[2])).groupByKey().map(lambda x: (x[0],sum(list(x[1]))/len(list(x[1])))).collectAsMap()
# {bid:avg_star}
hist_bid_avg_rate = hist_uid_bid_rate.map(lambda x: (x[1],x[2])).groupByKey().map(lambda x: (x[0],sum(list(x[1]))/len(list(x[1])))).collectAsMap()

# read test data and train and get basic info
test_data = sc.textFile(test_path)
test_head = test_data.first()
test_data = test_data.filter(lambda x: x!=test_head) #exclude the first line of name
# (bid,[uid,uid,...])
bid_uids_to_pred = test_data.map(lambda x: x.split(",")).map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y: x+y).mapValues(lambda x: [*set(x)])
# {bid:['uid,uid,...],...}
test_bid_uids_info_dict = bid_uids_to_pred.collectAsMap()


after_pred = bid_uids_to_pred.map(lambda x: predict(x[0],test_bid_uids_info_dict)).flatMap(lambda x: x)
res = after_pred.collect()

# less than 130 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, header: user_id, business_id, similarity
with open(output_path,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "business_id", "similarity"])
    for i in res:
        # print(i)
        writer.writerow(i)

# # calculate RMSE < 1.09
# # ((uid,bid),rate)
# after_pred = bid_uids_to_pred.map(lambda x: predict(x[0],test_bid_uids_info_dict)).flatMap(lambda x: x)
# # truth = test_data.map(lambda x: x.split(",")).map(lambda x: ((x[0],x[1]),float(x[2])))
# my_res = after_pred.map(lambda x: ((x[0],x[1]),x[2]))
# RMSE = my_res.join(truth).map(lambda x: (x[1][0]-x[1][1])**2).reduce(lambda x, y: x+y)
# print(math.sqrt(RMSE/my_res.count()))

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_1.py
# "../resource/asnlib/publicdata/yelp_train.csv"
# "../resource/asnlib/publicdata/yelp_val.csv"
# "./task2_1.csv"

