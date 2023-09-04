import sys
import random
import time
import numpy as np
import math
import copy
from pyspark import SparkContext
from sklearn.cluster import KMeans

def generate_DS_RS(sample,labels,current_ds,current_rs):
    res_rs = {}
    # {cluster_index,data_index}
    cluster_info = {}
    for i in range(len(labels)):
        if labels[i] not in cluster_info:
            cluster_info[labels[i]] = [list(sample.keys())[i]]
        else:
            cluster_info[labels[i]].append(list(sample.keys())[i])
        
    # gemerate ds and rs
    for cluster in cluster_info:
        members = cluster_info[cluster]
        # if contain only one point
        if len(members)<=1:
            res_rs[members[0]] = sample[members[0]]
            current_rs[members[0]] = sample[members[0]]
        else:
            features = [sample[i] for i in members]
            tmp = np.array(features)
            # generate statistics
            cluster_n = len(tmp)
            cluster_sum = tmp.sum(axis=0)
            cluster_sumsq = (tmp**2).sum(axis=0)
            current_ds[cluster] = [members,cluster_n,cluster_sum,cluster_sumsq]
    
    return current_ds,current_rs
    # return current_ds,current_rs,res_rs

def generate_CS_RS(current_cs,current_rs):
    # current_cs = 
    # if # of points in RS more than # of 5*n_cluster
    if len(current_rs) > 5*n_cluster:
        tmp = list(current_rs.values())
        model = KMeans(n_clusters=5*n_cluster,random_state=0).fit(tmp)
        labels = model.labels_
        current_cs,current_rs = generate_DS_RS(current_rs,labels,current_cs,current_rs)
    return current_cs,current_rs

# “The intermediate results”
# Round {𝑖}: the numbers in the order of “the number of the discard points”,
# “the number of the clusters in the compression set”,“the number of the compression points”,
# “the number of the points in the retained set”.
# Round {𝑖}: ds_len,cs_num,cs_len,rs_num

def intermidiate_res(current_ds,current_cs,current_rs):
    ds_members = [info[0] for info in current_ds.values()]
    ds_len = sum([len(member) for member in ds_members])
    
    cs_members = [info[0] for info in current_cs.values()]
    cs_num = len(current_cs)
    cs_len = sum([len(member) for member in cs_members])

    rs_num = len(current_rs)
    return (ds_len,cs_num,cs_len,rs_num)

def cal_ma_dist(cluster_centroid,cluster_statistics):
    cluster_n = cluster_statistics[1]
    cluster_sum = np.array(cluster_statistics[2])
    cluster_sumsq = np.array(cluster_statistics[3])
    centroid = cluster_sum/cluster_n
    sigma = np.sqrt((cluster_sumsq/cluster_n)-centroid**2)
    dist = np.sqrt((((cluster_centroid-centroid)/sigma)**2).sum())
    return dist

def merge_cs(current_cs):
    cs = copy.deepcopy(current_cs)
    merged = set()
    for index_1 in cs:
        info_1 = cs[index_1]
        length_1 = info_1[1]
        sum_1 = info_1[2]
        sumsq_1 = info_1[3]
        centroid_1 = sum_1/length_1
        dimension_1 = len(centroid_1)
        dist_min = 100000
        neareast_cluster = None
        # go through cs and find neareast cluster
        for index_2 in cs:
            if index_2 != index_1 and index_2 not in merged:
                dist_btw_cluster = cal_ma_dist(centroid_1,cs[index_2])
                if dist_btw_cluster<=dist_min:
                    dist_min = dist_btw_cluster
                    neareast_cluster = index_2
        # merge and update
        if dist_min <= 2*math.sqrt(dimension_1):
            cluster_info = current_cs[neareast_cluster]
            member_updated = cluster_info[0]+info_1[0]
            n_updated = cluster_info[1]+length_1
            sum_updated = cluster_info[2]+sum_1
            sumsq_updated = cluster_info[3]+sumsq_1
            current_cs[neareast_cluster] = [member_updated,n_updated,sum_updated,sumsq_updated]
            merged.add(index_1)
    # delete merged clusters from cs
    for cluster in merged:
        cs.pop(cluster)
    return cs

def assignToSets(sample,current_ds,current_cs,current_rs):
    for point in sample:
        features = np.array(sample[point])
        dimension = len(features)
        dist_min = 1000000
        cluster_assign_to = None

        for cluster in current_ds:
            dist = cal_ma_dist(features,current_ds[cluster])
            if dist<=dist_min:
                dist_min = dist
                cluster_assign_to = cluster
        
        # Step 8. compare using the Mahalanobis Distance and assign to the nearest DS clusters if the distance is<sprt(2𝑑)
        if dist_min <= 2*math.sqrt(dimension):
            cluster_info_ds = current_ds[cluster_assign_to]
            member_updated = cluster_info_ds[0]+[point]
            n_updated = cluster_info_ds[1]+1
            sum_updated = cluster_info_ds[2]+features
            sumsq_updated = cluster_info_ds[3]+(features**2)
            current_ds[cluster_assign_to] = [member_updated,n_updated,sum_updated,sumsq_updated]
        else: # Step 9. For that are not assigned to DS, ssign them to the nearest CS clusters if the distance is<sprt(2𝑑)
            dist_min_cs = 1000000
            cluster_assign_to = None

            for cluster in current_cs:
                dist = cal_ma_dist(features,current_cs[cluster])
                if dist<=dist_min_cs:
                    dist_min_cs = dist
                    cluster_assign_to = cluster
            if dist_min_cs <= 2*math.sqrt(dimension):
                cluster_info_cs = current_cs[cluster_assign_to]
                member_updated = cluster_info_cs[0]+[point]
                n_updated = cluster_info_cs[1]+1
                sum_updated = cluster_info_cs[2]+features
                sumsq_updated = cluster_info_cs[3]+(features**2)
                current_cs[cluster_assign_to] = [member_updated,n_updated,sum_updated,sumsq_updated]
            else:#  Step 10. For not assigned to a DS cluster or a CS cluster,assign them to RS.
                current_rs[point] = sample[point]
    # Step 11. Run K-Means on the RS with a large K to generate CS and RS 
    current_cs,current_rs = generate_CS_RS(current_cs,current_rs)
    # Step12.Merge CS clusters that have a Mahalanobis Distance<sprt(2𝑑)
    current_cs = merge_cs(current_cs)
    return current_ds,current_cs,current_rs

def final_merge(current_ds,current_cs):
    for cs in current_cs:
        cluster_info = current_cs[cs]
        cluster_n = cluster_info[1]
        cluster_sum = cluster_info[2]
        cluster_sumsq = cluster_info[3]
        centroid = cluster_sum/cluster_n
        dimension = len(centroid)
        dist_min = 100000
        neareast_cluster = None
        # go through ds and find neareast cluster
        for ds in current_ds:
            dist_btw_cluster = cal_ma_dist(centroid,current_ds[ds])
            if dist_btw_cluster<=dist_min:
                dist_min = dist_btw_cluster
                neareast_cluster = ds
        # merge and update
        if dist_min <= 2*math.sqrt(dimension):
            cluster_info = current_ds[neareast_cluster]
            member_updated = cluster_info[0]+cluster_info[0]
            n_updated = cluster_info[1]+cluster_n
            sum_updated = cluster_info[2]+cluster_sum
            sumsq_updated = cluster_info[3]+cluster_sumsq
            current_ds[neareast_cluster] = [member_updated,n_updated,sum_updated,sumsq_updated]
            current_cs.pop(cs)
    return current_ds,current_cs

# BFR
# task.py <input_path> <n_cluster> <output_path>
input_path = sys.argv[1]
n_cluster = int(sys.argv[2])
output_path = sys.argv[3]
# input_path = "../data/input/hw6_clustering.txt"
# n_cluster = 50
# output_path = "../data/output/task.txt"

s_time = time.time()
sc = SparkContext("local[*]",appName="task").getOrCreate()
sc.setLogLevel("ERROR")

# read data
data = sc.textFile(input_path)
# data point index,cluster index,features of data point
# [index,[features,...]]
data = data.map(lambda x: (int(x.split(",")[0]),[float(feature) for feature in x.split(",")[2:]]))

# The initialization of DS
DS = {}
CS = {}
RS = {}

# Step 1. Load 20% of the data randomly.
random.seed(553)
a = random.randint(1,round(time.time()/10000))
b = random.randint(1,round(time.time()/10000))
def hash_to_5_part(index):
    hash_value = ((a*index+b)%2023)%5
    return int(hash_value)

data = data.map(lambda x: (x[0],hash_to_5_part(x[0]),x[1]))
# random_num = random.randint(0,4)
random_num = 0
# {index,[features,...]}
sample = data.filter(lambda x: x[1]==random_num).map(lambda x: (x[0],x[2])).collectAsMap()

# Step 2. Run K-Means with a large K (e.g.,5 times of the number of the input clusters) on the data in memory 
# using the Euclidean distance as the similarity measurement.
kmeans = KMeans(n_clusters=n_cluster*5,random_state=0).fit(list(sample.values()))

# Step 3. 
# count labels 
labels = kmeans.labels_
# {index,cluster}
label_info = {}
# {cluster,cnt}
label_cnt = {}
for i in range(len(sample.keys())):
    label_info[list(sample.keys())[i]] = labels[i]
    if labels[i] not in label_cnt:
        label_cnt[labels[i]] = 1
    else:
        label_cnt[labels[i]] += 1

# move all the clusters that contain only one point to RS (outliers)
cluster_cnt_1 = []
for label in label_cnt:
    if label_cnt[label] == 1:
        cluster_cnt_1.append(label)

sample_2 = copy.deepcopy(sample)
for index in label_info:
    if label_info[index] in cluster_cnt_1:
        RS[index] = sample[index]
        sample_2.pop(index)

# Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
kmeans_2 = KMeans(n_clusters=n_cluster,random_state=0).fit(list(sample_2.values()))

# Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e.,discard their points and generate statistics).
labels = kmeans_2.labels_
DS,RS = generate_DS_RS(sample_2,labels,DS,RS)

# Step 6. Run K-Means on the points in the RS with a large K to generate CS and RS
CS,RS = generate_CS_RS(CS,RS)

res = intermidiate_res(DS,CS,RS)
final_res = []
final_res.append(res)

hash_parts = [0,1,2,3,4]
hash_parts.pop(random_num)
cnt = 0
# Step 7. Load another 20% of the data randomly.
for num in hash_parts:
    random_sample = data.filter(lambda x: x[1]==num).map(lambda x: (x[0],x[2])).collectAsMap()
    # step8-12: For the new points, compare and assign to DS,CS,RS
    DS,CS,RS = assignToSets(random_sample,DS,CS,RS)
    cnt += 1
    # print("round finished")
    # in last round, merge all cs and all rs into neareast cluster
    if cnt == 4:
        DS,CS = final_merge(DS,CS)
    res = intermidiate_res(DS,CS,RS)
    final_res.append(res)

# check Percentage of discard points after last round: >98.5%
# print(final_res[-1][0]/len(data.collect()))

# The clustering results
# data points index and their clustering results after the BFR algorithm
# clustering results should be in [0,the number of clusters),cluster of outliers should be represented as -1.
final_res_2 = []
for cluster in DS:
    for member in DS[cluster][0]:
        final_res_2.append((member,cluster))

if len(CS) != 0:
    for cluster in CS:
        for member in CS[cluster][0]:
            final_res_2.append((member,-1))

if len(RS) != 0:
    for member in RS:
        final_res_2.append((member,-1))

# sort by data index
final_res_2 = sorted(final_res_2,key=lambda x: x[0])

# compute the accuracy of your clustering results to the ground truth: 98.0%
# from sklearn.metrics import normalized_mutual_info_score
# ground_truth = np.loadtxt(input_path, delimiter=",")
# print(normalized_mutual_info_score(ground_truth[:,1], np.array(final_res_2)[:,1]))
# print(final_res)

# less than 600 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to txt,format:
with open(output_path,"w") as f:
    f.write("The intermediate results:\n")
    for i in range(len(final_res)):
        round = final_res[i]
        output = "Round{"+str(i+1)+"}: "+str(round[0])+","+str(round[1])+","+str(round[2])+","+str(round[3])+"\n"
        # print(output)
        f.write(output)
    # f.write("\n")
    f.write("\nThe clustering results:\n")
    for i in range(len(final_res_2)):
        output = str(final_res_2[i][0])+","+str(final_res_2[i][1])+"\n"
        f.write(output)

#export PYSPARK_PYTHON=python3.6                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit 
# --executor-memory 4G --driver-memory 4G 
# task.py "../resource/asnlib/publicdata/hw6_clustering.txt" 50 "./task.txt"