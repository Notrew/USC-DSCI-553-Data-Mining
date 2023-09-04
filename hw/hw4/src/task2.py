#!/usr/bin/env python
# coding: utf-8

import sys
from pyspark import SparkContext
import time
from itertools import combinations
from collections import defaultdict
import copy

# community detection based on GraphFrames
# task2.py <filter_threshold> <input_file_path> <output_file_path_1> <output_file_path_2>
threshold = int(sys.argv[1])
input_path = sys.argv[2]
output_path_1 = sys.argv[3]
output_path_2 = sys.argv[4]
# threshold = 5
# input_path = "../data/input/ub_sample_data.csv"
# output_path_1 = "../data/output/betweenness.txt"
# output_path_2 = "../data/output/community.txt"

s_time = time.time()
sc = SparkContext("local[*]",appName="task1").getOrCreate()
sc.setLogLevel("ERROR")

# read data and exclude the first line of name
data = sc.textFile(input_path).filter(lambda x: x!="user_id,business_id") 
uid_bids = data.map(lambda x: x.split(",")).map(lambda x: (x[0],[x[1]])).reduceByKey(lambda x,y: x+y).mapValues(lambda x: set(x))
# filter first time and construct {uid:[bid,bid,...]}
uid_bids_dict = uid_bids.filter(lambda x: len(x[1])>=threshold).collectAsMap()

# construct nodes and edges
# node-->user, edge-->exist if two nodes" common bids >= threshold

# get user pairs
uids = list(uid_bids_dict.keys())
user_pairs = combinations(uids,2)
# then caculate len(common_bids) and select those cnt>=threshold
valid_pairs_len = []
for i in user_pairs:
    len_common_bids = len(uid_bids_dict[i[0]].intersection(uid_bids_dict[i[1]]))
    if len_common_bids>=threshold:
        valid_pairs_len.append((sorted(i),len_common_bids))

nodes = set()
for pair in valid_pairs_len:
    nodes.add(tuple(pair[0])[0])
    nodes.add(tuple(pair[0])[1])

connections = defaultdict(set)
for pairs_len in valid_pairs_len:
    pairs = pairs_len[0]
    connections[pairs[0]].add(pairs[1])
    connections[pairs[1]].add(pairs[0])

# Girvan-Newman Alg
# visit each node X once (BFS)
# compute the # of the shortest paths from X to each of the other nodes
# repeat:
    # calculate betweenness of edges, and remove high betweennedd edges

def bfs(root,connections):
    parents_lst = defaultdict(list)
    depth = defaultdict(int)
    num_shortest_path = defaultdict(int)
    queue = []
    bfs_queue = []

    # set default value
    parents_lst[root] = None
    depth[root] = 0
    num_shortest_path[root] = 1
    bfs_queue.append(root)

    # prepare children of root
    for adjacent in connections[root]:
        parents_lst[adjacent] = [root]
        depth[adjacent] = 1
        num_shortest_path[adjacent] = 1
        bfs_queue.append(adjacent)
        queue.append(adjacent)

    while queue:
        cur_node = queue.pop(0)
        # go through neighbours
        for adjacent in connections[cur_node]:
            # if it didn't appear before, set it as cur_node's child
            if adjacent not in bfs_queue:
                parents_lst[adjacent] = [cur_node]
                depth[adjacent] = depth[cur_node]+1
                bfs_queue.append(adjacent)
                queue.append(adjacent)
            # it appeared before
            else:
                if depth[adjacent]==depth[cur_node]+1:
                    parents_lst[adjacent].append(cur_node)
        num_shortest_path[cur_node] = sum(num_shortest_path[parent] for parent in parents_lst[cur_node])
    bfs_queue.reverse()
    return bfs_queue,parents_lst,num_shortest_path

def cal_credit(bfs_queue_rever,parents_lst,num_shortest_path):
    # set default credit
    basic_credit = {}
    # credit of root = 0
    basic_credit[bfs_queue_rever[-1]] = 0
    # else = 1 at beginning
    for node in bfs_queue_rever[:-1]:
        basic_credit[node] = 1

    credit_dict = {}
    # form bottom to
    for child in bfs_queue_rever[:-1]:
        for parent in parents_lst[child]:
            weight = num_shortest_path[parent]/num_shortest_path[child]
            credit = basic_credit[child]*weight
            basic_credit[parent] += credit
            credit_dict[tuple(sorted((child,parent)))] = credit

    return [(pair,credit) for pair,credit in credit_dict.items()]

def GN_Alg(root,connections):
    bfs_res = bfs(root,connections)
    credit_res = cal_credit(bfs_res[0],bfs_res[1],bfs_res[2])
    return credit_res

# calculate betweenness
betweenness = sc.parallelize(nodes).map(lambda x: GN_Alg(x,connections)).flatMap(lambda x: x).reduceByKey(lambda x,y: x+y).map(lambda x: (x[0],x[1]/2))
betweenness = betweenness.map(lambda x: (x[0],round(x[1],5))).collect()
betweenness = sorted(betweenness,key=lambda x: (-x[1],x[0][0],x[0][1]))

# write to txt, format: (uid1, uid2), betweenness
with open(output_path_1,"w") as f:
    for i in betweenness:
        output = "('"+i[0][0]+"', '"+i[0][1]+"'),"+str(i[1])+"\n"
        f.write(output)

# detect communities
def min_cut(pair,connections):
    p1 = pair[0]
    p2 = pair[1]
    # update the connections
    # print(p1,p2)
    connections[p1].remove(p2)
    connections[p2].remove(p1)

def get_community(all_nodes,connections):
    communities = []
    queue = []
    nodes_visited = []
    for node in all_nodes:
        if node not in nodes_visited:
            tmp = [node]
            queue.append(node)
            while queue:
                cur_node = queue.pop(0)
                for adjacent in connections[cur_node]:
                    if adjacent not in tmp:
                        tmp.append(adjacent)
                        queue.append(adjacent)
            tmp.sort()
            nodes_visited += tmp
            communities.append(tmp)
    return communities

def cal_modularity(communities):
    org_modularity = 0
    for community in communities:
        for i in community:
            for j in community:
                actual = 1 if j in connections[i] else 0
                expected = (degree_info[i]*degree_info[j])/(2*m)
                org_modularity += (actual - expected)
    return org_modularity/(2*m)

# num of edges
m = len(valid_pairs_len)
# degree
degree_info = {}
for node in connections:
    degree_info[node] = len(connections[node])

# if still has edges can cut
connections_info = copy.deepcopy(connections)
total_modularity = -1
remaining_edges = m
while remaining_edges>0:
    highest_betweenness = betweenness[0][1]
    # cut
    for pair_betweenness in betweenness:
        if pair_betweenness[1] == highest_betweenness:
            # print(pair_betweenness[0])
            min_cut(pair_betweenness[0],connections_info)
            # update remaining_edges
            remaining_edges -= 1
    # update
    community_after_cut = get_community(nodes,connections_info)
    modularity = cal_modularity(community_after_cut)
    if modularity>total_modularity:
        # update modularity
        total_modularity = modularity
        communities_res = copy.deepcopy(community_after_cut)
    # recalculate betweenness
    betweenness = sc.parallelize(nodes).map(lambda x: GN_Alg(x,connections_info)).flatMap(lambda x: x).reduceByKey(lambda x,y: x+y)\
        .map(lambda x: (sorted(x[0]),x[1]/2)).sortBy(lambda x: (-x[1],x[0][0],x[0][1])).collect()

res = sc.parallelize(communities_res).map(lambda x: sorted(x)).sortBy(lambda x: (len(x),x)).collect()

with open(output_path_2, "w") as f:
    for i in res:
        f.write(str(i).strip("[").strip("]")+"\n")

# less than 400 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit 
# --executor-memory 4G --driver-memory 4G 
# task2.py 2 "../resource/asnlib/publicdata/ub_sample_data.csv" "./task2_1.txt" "./task2_2.txt"

