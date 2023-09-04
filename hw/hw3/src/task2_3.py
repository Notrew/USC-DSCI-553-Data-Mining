#!/usr/bin/env python
# coding: utf-8

import sys
# import os
from pyspark import SparkContext
import time
import math
import json
import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn import preprocessing
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error

# task2_3.py <folder_path> <test_file_path> <output_file_path>
folder_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
# folder_path = "../data/input/"
# test_path = "../data/input/yelp_val.csv"
# output_path = "../data/output/task2_3.csv"

train_path = folder_path+"yelp_train.csv"
user_path = folder_path+"user.json"
business_path = folder_path+"business.json"
review_train_path = folder_path+"review_train.json"
# checkin_path = folder_path+"checkin.json"
# tip_path = folder_path+"tip.json"
photo_path = folder_path+"photo.json"

s_time = time.time()
sc = SparkContext("local[*]",appName="task2_3").getOrCreate()

# hybrid = item-based+model-based
# item-based
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
    
# read train data and train and get basic info
# split one row into (uid,bid,star)
train_data = sc.textFile(train_path)
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

user_id = []
business_id = []
pred_item = []
for i in res:
    user_id.append(i[0])
    business_id.append(i[1])
    pred_item.append(i[2])
res_item = pd.DataFrame({"user_id":user_id, "business_id":business_id, "prediction":pred_item})

# truth = pd.read_csv(test_path)
# merged_item = truth.merge(res_item,on=["user_id", "business_id"])
# print(mean_squared_error(merged_item["stars"],merged_item["prediction"],squared=False))

# model-based
def getPriceRange(attributes,key):
    if attributes:
        if key in attributes.keys():
            return float(attributes.get(key))
    return 0

# var_rate and photo_cnt may be None
def fillInNone(num,default):
    if num:
        return num
    else:
        return default
    
# uid_info: {user_id:(review_count,fans,average_stars,friends,social,var_rate)}
# bid_info: {business_id:(stars,review_count,price_range,var_rate,phtot_cnt)}
# get and join features together, all inputs are dictionaries
def mergrFeatures(df_org,uid_info,bid_info):
    col_names = ["user_review_cnt","user_fans","user_avg_rate","user_var_rate",
                 "user_friends","user_social","user_year","user_elite","user_compliment",
                 "bsn_avg_rate","bsn_var_rate","bsn_review_cnt","bsn_price_range","bsn_photo_cnt"]
    user_review_cnt = []
    user_fans = []
    user_avg_rate = []
    user_var_rate = []
    user_friends = []
    user_social = []
    user_year = []
    user_elite = []
    user_compliment = []
    bsn_avg_rate = []
    bsn_var_rate = []
    bsn_review_cnt = []
    bsn_price_range = [ ]
    bsn_photo_cnt = []
    for uid in df_org["user_id"]:
        if uid in uid_info.keys():
            user_review_cnt.append(uid_info.get(uid)[0])
            user_fans.append(uid_info.get(uid)[1])
            user_avg_rate.append(uid_info.get(uid)[2])
            user_friends.append(uid_info.get(uid)[3])
            user_social.append(uid_info.get(uid)[4])
            user_year.append(uid_info.get(uid)[5])
            user_elite.append(uid_info.get(uid)[6])
            user_compliment.append(uid_info.get(uid)[7])
            user_var_rate.append(uid_info.get(uid)[8])
        else:
            user_review_cnt.append(uid_review_cnt_whole)
            user_fans.append(uid_fans_whole)
            user_avg_rate.append(uid_avg_rate_whole)
            user_friends.append(uid_fri_whole)
            user_social.append(uid_social_whole)
            user_year.append(0)
            user_elite.append(0)
            user_compliment.append(0)
            user_var_rate.append(0)
    for bid in df_org["business_id"]:
        if bid in bid_info.keys():
            bsn_avg_rate.append(bid_info.get(bid)[0])
            bsn_var_rate.append(bid_info.get(bid)[3])
            bsn_review_cnt.append(bid_info.get(bid)[1])
            bsn_price_range.append(bid_info.get(bid)[2])
            bsn_photo_cnt.append(bid_info.get(bid)[4])
        else:
            bsn_avg_rate.append(bid_avg_rate_whole)
            bsn_review_cnt.append(bid_review_cnt_whole)
            bsn_price_range.append(bid_price_range_whole)
            bsn_var_rate.append(0)
            bsn_photo_cnt.append(0)
            # bsn_avg_rate.append(3)
            # bsn_review_cnt.append(0)
            # bsn_price_range.append(0)
    for i in col_names:
        df_org[i] = locals()[i]
    return df_org

# read and pre_datasets, select features
# train_data
train_data = sc.textFile(train_path)
head = train_data.first()
train_data = train_data.filter(lambda x: x!=head) #exclude the first line of name
train_uid_bid_rate = train_data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1],float(x[2])))
# hist_uids = hist_uid_bid_rate.map(lambda x: x[0]).distinct()
# hist_bids = hist_uid_bid_rate.map(lambda x: x[1]).distinct()

# user, select: user_id,(review_count,fans,average_stars,friends,useful+funny+cool,year,elite,compliment))
user = sc.textFile(user_path).map(lambda x: json.loads(x))
user = user.map(lambda x: (x["user_id"],(x["review_count"],x["fans"],x["average_stars"],len(x["friends"].split(",")),
                                         x["useful"]+x["funny"]+x["cool"],(2023-int(x["yelping_since"][:4])),len(x["elite"].split(",")),
                                         x["compliment_hot"]+x["compliment_more"]+x["compliment_profile"]+x["compliment_cute"]+\
                                         x["compliment_list"]+x["compliment_note"]+x["compliment_plain"]+x["compliment_cool"]+\
                                         x["compliment_funny"]+x["compliment_writer"]+x["compliment_photos"]
                                         )))

# business, select: business_id,(stars,review_count,attributes[RestaurantsPriceRange2])) 
# try to add attributes[OutdoorSeating,RestaurantsDelivery,RestaurantsGoodForGroups,RestaurantsReservations,RestaurantsTakeOut] later, True/False
business = sc.textFile(business_path).map(lambda x: json.loads(x))
business = business.map(lambda x: (x["business_id"],(x["stars"],x["review_count"],getPriceRange(x["attributes"],"RestaurantsPriceRange2"))))

# review_train, (user_id,business_id,stars)
review_train = sc.textFile(review_train_path).map(lambda x: json.loads(x))
review_train = review_train.map(lambda x: (x["user_id"],x["business_id"],x["stars"]))

# photo, select:business_id,label(['food', 'drink', 'outside', 'inside', 'menu'])
photo = sc.textFile(photo_path).map(lambda x: json.loads(x))
photo = photo.map(lambda x: (x["business_id"],x["label"]))

# aggragation
# user, select: user_id,(review_count,fans,average_stars,friends,useful,funny,cool))
# review_cnt
# if uid not in extra dataset, use the average review_cnt in extra dataset 
uid_review_cnt_whole = user.map(lambda x: x[1][0]).mean()
# fans
uid_fans_whole = user.map(lambda x: x[1][1]).mean()
# avg_rate
uid_avg_rate_whole = user.map(lambda x: x[1][2]).mean()
# friends
uid_fri_whole = user.map(lambda x: x[1][3]).mean()
# (usrful+funny+cool)
uid_social_whole = user.map(lambda x: x[1][4]).mean()
# var_rate
uid_var_rate = review_train.map(lambda x: (x[0],x[2])).groupByKey().mapValues(lambda x: np.var((list(x))))
# uid_info: {user_id:(review_count,fans,average_stars,friends,social,year,elite,compliment,var_rate)}
uid_info = user.leftOuterJoin(uid_var_rate).map(lambda x: x).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).collectAsMap()

# business, select: business_id,(stars,review_count,attributes[RestaurantsPriceRange2])) 
# avg_rate
bid_avg_rate_whole = business.map(lambda x: x[1][0]).mean()
# var_rate
bid_var_rate = review_train.map(lambda x: (x[1],x[2])).groupByKey().mapValues(lambda x: np.var((list(x))))
# review_cnt
bid_review_cnt_whole = business.map(lambda x: x[1][1]).mean()
# price_range
bid_price_range_whole = business.map(lambda x: x[1][2]).mean()
# photo_cnt
bid_photo_cnt = photo.filter(lambda x: x[1]!="menu").map(lambda x: (x[0],1)).reduceByKey(lambda x,y:x+y)
# bid_info: {business_id:(stars,review_count,price_range,var_rate,phtot_cnt)}
bid_info = business.leftOuterJoin(bid_var_rate).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).\
                    leftOuterJoin(bid_photo_cnt).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).collectAsMap()

df_train_org = pd.DataFrame(train_uid_bid_rate.collect(),columns=["user_id","business_id","stars"])
df_train = mergrFeatures(df_train_org,uid_info,bid_info)
x_train = df_train.drop(["user_id","business_id","stars"],axis=1)
# scaler = preprocessing.StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# standarize
for col in x_train.columns:
    x_train[col] = (x_train[col]-x_train[col].mean())/x_train[col].std()
y_train = df_train["stars"]

# read test data and train and get basic info
test_data = sc.textFile(test_path)
test_head = test_data.first()
test_data = test_data.filter(lambda x: x!=test_head) #exclude the first line of name
uid_bid_to_pred = test_data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1])).collect()
df_test_org = pd.DataFrame(uid_bid_to_pred,columns=["user_id","business_id"])
df_test = mergrFeatures(df_test_org,uid_info,bid_info)
x_test = df_test.drop(["user_id","business_id"],axis=1)
# x_test = scaler.transform(x_test)
for col in x_test.columns:
    x_test[col] = (x_test[col]-x_test[col].mean())/x_test[col].std()

# fit model
xgb_model = xgb.XGBRegressor(alpha=0.5,learning_rate=0.05,colsample_bytree=0.4,max_depth=7,n_estimators=200,subsample=0.6,random_state=0)
xgb_model.fit(x_train.drop(["bsn_price_range","bsn_var_rate","user_var_rate"],axis=1),y_train)
# xgb_model.fit(x_train,y_train)

# predict
y_pred = xgb_model.predict(x_test.drop(["bsn_price_range","bsn_var_rate","user_var_rate"],axis=1))
# y_pred = xgb_model.predict(x_test)

res_model = pd.DataFrame({"user_id":[x[0] for x in uid_bid_to_pred],"business_id":[x[1] for x in uid_bid_to_pred],"prediction": y_pred})
# truth = pd.read_csv(test_path)
# merged_model = truth.merge(res_model,on=["user_id", "business_id"])
# print(mean_squared_error(merged_model["stars"],res_model["prediction"],squared=False))

# # combine two methods
# res_merged = res_model.merge(res_item,on=["user_id","business_id"])
# for a in np.linspace(0.8,1,20):
#     merged = []
#     for i in range(res_merged.shape[0]):
#         merged.append(res_merged["prediction_x"][i]*a+res_merged["prediction_y"][i]*(1-a))
#     res_merged["final_res"] = merged
#     print(a,mean_squared_error(truth["stars"],res_merged["final_res"],squared=False))

# combine two methods
res_merged = res_model.merge(res_item,on=["user_id","business_id"])
merged = []
a = 0.97
for i in range(res_merged.shape[0]):
    merged.append(res_merged["prediction_x"][i]*a+res_merged["prediction_y"][i]*(1-a))
res_merged["final_res"] = merged

# calculate RMSE < 1.00
# from sklearn.metrics import mean_squared_error
# merged = truth.merge(res_merged,on=["user_id", "business_id"])
# print(mean_squared_error(merged_item["stars"],merged_item["prediction"],squared=False))
# print(mean_squared_error(merged_model["stars"],merged_model["prediction"],squared=False))
# print(mean_squared_error(merged["stars"],merged["final_res"],squared=False))

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, header: user_id, business_id, prediction
output= res_merged[["user_id","business_id","final_res"]]
output.to_csv(output_path,index=False)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_2.py
# "../resource/asnlib/publicdata/"
# "../resource/asnlib/publicdata/yelp_val.csv"
# "./task2_2.csv"

