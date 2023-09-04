#!/usr/bin/env python
# coding: utf-8

import sys
import os
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

# task2_2.py <folder_path> <test_file_path> <output_file_path>
# folder_path = sys.argv[1]
# test_path = sys.argv[2]
# output_path = sys.argv[3]
folder_path = "../data/input/"
test_path = "../data/input/yelp_val_in.csv"
output_path = "../data/output/task2_2.csv"

train_path = folder_path+"yelp_train.csv"
user_path = folder_path+"user.json"
business_path = folder_path+"business.json"
review_train_path = folder_path+"review_train.json"
# checkin_path = folder_path+"checkin.json"
# tip_path = folder_path+"tip.json"
photo_path = folder_path+"photo.json"

s_time = time.time()
sc = SparkContext("local[*]",appName="task2_2").getOrCreate()

# model-based CF recommendation system with Pearson similarity
# step
    # train_data+features join together
    # fit model 
    # select parameters
# predict on test dataset

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
                 "user_friends","user_social","bsn_avg_rate","bsn_var_rate",
                "bsn_review_cnt","bsn_price_range","bsn_photo_cnt"]
    user_review_cnt = []
    user_fans = []
    user_avg_rate = []
    user_var_rate = []
    user_friends = []
    user_social = []
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
            user_var_rate.append(uid_info.get(uid)[5])
        else:
            user_review_cnt.append(uid_review_cnt_whole)
            user_fans.append(uid_fans_whole)
            user_avg_rate.append(uid_avg_rate_whole)
            user_friends.append(uid_fri_whole)
            user_social.append(uid_social_whole)
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

# user, select: user_id,(review_count,fans,average_stars,friends,useful,funny,cool))
user = sc.textFile(user_path).map(lambda x: json.loads(x))
user = user.map(lambda x: (x["user_id"],(x["review_count"],x["fans"],x["average_stars"],len(x["friends"].split(",")),x["useful"]+x["funny"]+x["cool"])))

# business, select: business_id,(stars,review_count,attributes[RestaurantsPriceRange2])) 
# try to add attributes[OutdoorSeating,RestaurantsReservations,RestaurantsTakeOut] later, True/False
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
# uid_info: {user_id:(review_count,fans,average_stars,friends,social,var_rate)}
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

# uid_info: {user_id:(review_count,fans,average_stars,friends,social,var_rate)}
# uid_info.get("QPREECpJrp8Dj3_TK22oqg")
# bid_info: {business_id:(stars,review_count,price_range,var_rate,phtot_cnt)}
# bid_info.get("ugLqbAvBdRDc-gS4hpslXw")

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
# y_true = test_data.map(lambda x: x.split(",")).map(lambda x: float(x[2])).collect()

# # select parameters
# xgb_model = xgb.XGBRegressor()
# param_grid = {"alpha":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
#             #   "max_depth":range(3,10,2),
#               "learning_rate":[0.05,0.01,0.1]}
# grid_search = GridSearchCV(xgb_model,param_grid,cv=5)
# grid_search = grid_search.fit(x_train,y_train)
# alpha = grid_search.best_params_["alpha"]
# learning_rate = grid_search.best_params_["learning_rate"]
# print(alpha,learning_rate)

# fit model
alpha= 0.6
learning_rate = 0.05
xgb_model = xgb.XGBRegressor(alpha=alpha,learning_rate=learning_rate,random_state=0)
xgb_model.fit(x_train,y_train)
# print(mean_squared_error(y_train,xgb_model.predict(x_train),squared=False))

# predict
y_pred = xgb_model.predict(x_test)
output = pd.DataFrame({"user_id":[x[0] for x in uid_bid_to_pred],"business_id":[x[1] for x in uid_bid_to_pred],"prediction": y_pred})
# calculate RMSE < 1.00
# print(mean_squared_error(np.array(y_true),y_pred,squared=False))

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, header: user_id, business_id, prediction
output.to_csv(output_path,index=False)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task2_2.py
# "../resource/asnlib/publicdata/"
# "../resource/asnlib/publicdata/yelp_val_in.csv"
# "./task2_2.csv"

