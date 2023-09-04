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

# Methood description:
# About the model, I still use the xgboost regressor, but with more features and some methods to tune parameters and select features.
# At first, I extracted features from 6 files and put them all into the data frame. 
# Then to improve accuracy, I ran RFECV and observe the property of each feature to select the best feature combination. 
# Next, use these selected features as constants to tune parameters, first I ran grid search method to choose the best pair of alpha and learning rate. 
# And in order to save time, ran loops to select the best max_depth, subsample, and colsample_bytree once completed the selection of alpha and learning rate.


# on validation set
# Error Distribution:
# >=0 and <1: 102240
# >=1 and <2: 32790
# >=2 and <3: 6179
# >=3 and <4: 834
# >=4: 1

# RMSE:
# 0.9796350306200236

# Execution time:
# 90.9462993144989


# task2_2.py <folder_path> <test_file_path> <output_file_path>
folder_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
# folder_path = "../data/input/"
# test_path = "../data/input/yelp_val_in.csv"
# output_path = "../data/output/task2_2.csv"

train_path = folder_path+"yelp_train.csv"
user_path = folder_path+"user.json"
business_path = folder_path+"business.json"
review_train_path = folder_path+"review_train.json"
photo_path = folder_path+"photo.json"
checkin_path = folder_path+"checkin.json"
tip_path = folder_path+"tip.json"

sc = SparkContext("local[*]",appName="competition").getOrCreate()
sc.setLogLevel("ERROR")

def getPriceRange(attributes,key):
    if attributes:
        if key in attributes.keys():
            return float(attributes.get(key))
    return 0

def convertBi(attributes,key):
    if attributes and key in attributes.keys():
        if attributes[key] != "False":
            return 1
        else:
            return 0
    else:
        return 0
        
# var_rate and photo_cnt may be None
def fillInNone(num,default):
    if num:
        return num
    else:
        return default
    
# get and join features together, all inputs are dictionaries
def mergrFeatures(df_org,uid_info,bid_info):
    col_names = ["user_review_cnt","user_fans","user_avg_rate","user_var_rate","user_friends","user_social","user_year","user_elite","user_compliment","user_tip",
                 "bsn_avg_rate","bsn_var_rate","bsn_review_cnt","bsn_price_range","bsn_photo_cnt","bsn_attribute_score","bsn_checkin","bsn_tip"]
    user_review_cnt = []
    user_fans = []
    user_avg_rate = []
    user_var_rate = []
    user_friends = []
    user_social = []
    user_year = []
    user_elite = []
    user_compliment = []
    user_tip = []
    
    bsn_avg_rate = []
    bsn_var_rate = []
    bsn_review_cnt = []
    bsn_price_range = [ ]
    bsn_photo_cnt = []
    bsn_attribute_score = []
    bsn_checkin = []
    bsn_tip = []

    # uid_info: {user_id:(review_count,fans,average_stars,friends,social,year,elite,compliment,var_rate,tips)}
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
            user_tip.append(uid_info.get(uid)[9])
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
            user_tip.append(0)
    # bid_info: {business_id:(stars,review_count,price_range,accribute_score,var_rate,phtot_cnt,checkin_time,tip)}
    for bid in df_org["business_id"]:
        if bid in bid_info.keys():
            bsn_avg_rate.append(bid_info.get(bid)[0])
            bsn_review_cnt.append(bid_info.get(bid)[1])
            bsn_price_range.append(bid_info.get(bid)[2])
            bsn_attribute_score.append(bid_info.get(bid)[3])
            bsn_var_rate.append(bid_info.get(bid)[4])
            bsn_photo_cnt.append(bid_info.get(bid)[5])
            bsn_checkin.append(bid_info.get(bid)[6])
            bsn_tip.append(bid_info.get(bid)[7])

        else:
            bsn_avg_rate.append(bid_avg_rate_whole)
            bsn_review_cnt.append(bid_review_cnt_whole)
            bsn_price_range.append(bid_price_range_whole)
            bsn_attribute_score.append(0)
            bsn_var_rate.append(0)
            bsn_photo_cnt.append(0)
            bsn_checkin.append(0)
            bsn_tip.append(0)
            # bsn_avg_rate.append(3)
            # bsn_review_cnt.append(0)
            # bsn_price_range.append(0)
    for i in col_names:
        df_org[i] = locals()[i]
    return df_org

def assignError(pred,truth,length):
    from_0_to_1 = 0
    from_1_to_2 = 0
    from_2_to_3 = 0
    from_3_to_4 = 0
    from_4_to_5 = 0
    for i in range(length):
        diff = abs(truth[i]-pred[i])
        if diff >= 0 and diff < 1:
            from_0_to_1 += 1
        elif diff >= 1 and diff < 2:
            from_1_to_2 += 1
        elif diff >= 2 and diff < 3:
            from_2_to_3 += 1
        elif diff >= 3 and diff < 4:
            from_3_to_4 += 1
        elif diff >= 4:
            from_4_to_5 += 1
    return [from_0_to_1,from_1_to_2,from_2_to_3,from_3_to_4,from_4_to_5]
# read and pre_datasets, select features
# train_data
train_data = sc.textFile(train_path)
head = train_data.first()
train_data = train_data.filter(lambda x: x!=head) #exclude the first line of name
train_uid_bid_rate = train_data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1],float(x[2])))

# user, select: user_id,(review_count,fans,average_stars,friends,useful+funny+cool,year,elite,compliment))
user = sc.textFile(user_path).map(lambda x: json.loads(x))
user = user.map(lambda x: (x["user_id"],(x["review_count"],x["fans"],x["average_stars"],len(x["friends"].split(",")),x["useful"]+x["funny"]+x["cool"],
                                         (2023-int(x["yelping_since"][:4])),len(x["elite"].split(",")),x["compliment_hot"]+x["compliment_more"]+\
                                         x["compliment_profile"]+x["compliment_cute"]+x["compliment_list"]+x["compliment_note"]+x["compliment_plain"]+\
                                         x["compliment_cool"]+x["compliment_funny"]+x["compliment_writer"]+x["compliment_photos"]
                                         )))

# business, select: business_id,(stars,review_count,priceRange,accribute_score)) 
business = sc.textFile(business_path).map(lambda x: json.loads(x))
business = business.map(lambda x: (x["business_id"],(x["stars"],x["review_count"],getPriceRange(x["attributes"],"RestaurantsPriceRange2"),
                                                    convertBi(x["attributes"],"OutdoorSeating")+convertBi(x["attributes"],"RestaurantsDelivery")+\
                                                    convertBi(x["attributes"],"RestaurantsGoodForGroups")+convertBi(x["attributes"],"RestaurantsReservations")+\
                                                    convertBi(x["attributes"],"RestaurantsTakeOut")+convertBi(x["attributes"],"BikeParking")+\
                                                    convertBi(x["attributes"],"BusinessAcceptsCreditCards")+convertBi(x["attributes"],"GoodForKids")+\
                                                    convertBi(x["attributes"],"HasTV")
                                                        )))

# review_train, (user_id,business_id,stars)
review_train = sc.textFile(review_train_path).map(lambda x: json.loads(x))
review_train = review_train.map(lambda x: (x["user_id"],x["business_id"],x["stars"]))

# photo, select:business_id,label(['food', 'drink', 'outside', 'inside', 'menu'])
photo = sc.textFile(photo_path).map(lambda x: json.loads(x))
photo = photo.map(lambda x: (x["business_id"],x["label"]))

# check_in, (business_id,checkin_time)
checkin = sc.textFile(checkin_path).map(lambda x: json.loads(x))
checkin = checkin.map(lambda x: (x["business_id"],len(x["time"])))

# tip, (user_id,business_id,likes)
tip = sc.textFile(tip_path).map(lambda x: json.loads(x))
tip = tip.map(lambda x: (x["user_id"],x["business_id"],x["likes"]))

# aggragation
# user, select: user_id,(review_count,fans,average_stars,friends,useful+funny+cool,year,elite,compliment))
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
# tip
uid_tips = tip.map(lambda x: (x[0],x[2])).groupByKey().mapValues(lambda x: sum(list(x)))
# uid_info: {user_id:(review_count,fans,average_stars,friends,social,year,elite,compliment,var_rate,tips)}
uid_info = user.leftOuterJoin(uid_var_rate).map(lambda x: x).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).\
    leftOuterJoin(uid_tips).map(lambda x: x).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).collectAsMap()

# business, select: business_id,(stars,review_count,priceRange,accribute_score)) 

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
# tip
bid_tips = tip.map(lambda x: (x[1],x[2])).groupByKey().mapValues(lambda x: sum(list(x)))
# bid_info: {business_id:(stars,review_count,price_range,accribute_score,var_rate,phtot_cnt,checkin_time,tip)}
bid_info = business.leftOuterJoin(bid_var_rate).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).\
                    leftOuterJoin(bid_photo_cnt).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).\
                    leftOuterJoin(checkin).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).\
                    leftOuterJoin(bid_tips).map(lambda x: x).map(lambda x: (x[0],x[1][0]+(fillInNone(x[1][1],0),))).collectAsMap()

df_train_org = pd.DataFrame(train_uid_bid_rate.collect(),columns=["user_id","business_id","stars"])
df_train = mergrFeatures(df_train_org,uid_info,bid_info)
x_train = df_train.drop(["user_id","business_id","stars"],axis=1)
# scaler = preprocessing.StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# standarize
# for col in x_train.columns:
#     x_train[col] = (x_train[col]-x_train[col].mean())/x_train[col].std()
y_train = df_train["stars"]

s_time = time.time()
# read test data and train and get basic info
test_data = sc.textFile(test_path)
test_head = test_data.first()
test_data = test_data.filter(lambda x: x!=test_head) #exclude the first line of name
uid_bid_to_pred = test_data.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1])).collect()
df_test_org = pd.DataFrame(uid_bid_to_pred,columns=["user_id","business_id"])
df_test = mergrFeatures(df_test_org,uid_info,bid_info)
x_test = df_test.drop(["user_id","business_id"],axis=1)
# x_test = scaler.transform(x_test)
# for col in x_test.columns:
#     x_test[col] = (x_test[col]-x_test[col].mean())/x_test[col].std()

# print(df_train.std().sort_values())

# # select parameters
# y_true = test_data.map(lambda x: x.split(",")).map(lambda x: float(x[2])).collect()
# res = 0.981
# for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
#     for lr in [0.05,0.01,0.1]:
# for max_depth in [3,4,5,6,7,8]:
#     for subsample in [0.5,0.6,0.7,0.8]:
#         for colsample_bytree in [0.5,0.6,0.7,0.8]:
#             xgb_model = xgb.XGBRegressor(alpha=0.8,learning_rate=0.1,max_depth=max_depth,subsample=subsample,colsample_bytree=colsample_bytree,random_state=0)
#             xgb_model.fit(x_train.drop(["bsn_var_rate","user_var_rate","bsn_price_range"],axis=1),y_train)
#             y_pred = xgb_model.predict(x_test.drop(["bsn_var_rate","user_var_rate","bsn_price_range"],axis=1))
#             rmse = mean_squared_error(np.array(y_true),y_pred,squared=False)
#             if rmse <= res:
#                 res = rmse
#                 print((max_depth,subsample,colsample_bytree),res)


# fit model
# xgb_model = xgb.XGBRegressor(alpha=0.5,learning_rate=0.05,colsample_bytree=0.4,max_depth=7,n_estimators=200,subsample=0.6,random_state=0)
xgb_model = xgb.XGBRegressor(alpha=0.8,learning_rate=0.1,colsample_bytree=0.4,max_depth=8,n_estimators=200,subsample=0.8,random_state=0)
xgb_model.fit(x_train.drop(["bsn_var_rate","user_var_rate","bsn_price_range"],axis=1),y_train)
# xgb_model.fit(x_train,y_train)

# predict
y_pred = xgb_model.predict(x_test.drop(["bsn_var_rate","user_var_rate","bsn_price_range"],axis=1))
# y_pred = xgb_model.predict(x_test)
output = pd.DataFrame({"user_id":[x[0] for x in uid_bid_to_pred],"business_id":[x[1] for x in uid_bid_to_pred],"prediction": y_pred})

# from sklearn.metrics import mean_squared_error
# y_true = test_data.map(lambda x: x.split(",")).map(lambda x: float(x[2])).collect()
# # calculate RMSE < 0.98
# print(math.sqrt(mean_squared_error(np.array(y_true),y_pred)))
# print(assignError(y_pred,np.array(y_true),len(y_pred)))

# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=xgb_model,step=1,scoring='neg_root_mean_squared_error')
# rfecv.fit(x_train,y_train)
# import matplotlib.pyplot as plt
# print('Optimal number of features :', rfecv.n_features_)
# print('Best features :', x_train.columns[rfecv.support_])
# print('Original features :', x_train.columns)
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score \n of number of selected features")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, header: user_id, business_id, prediction
output.to_csv(output_path,index=False)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G competition.py "../resource/asnlib/publicdata/" "../resource/asnlib/publicdata/yelp_val.csv" "./res.csv"

