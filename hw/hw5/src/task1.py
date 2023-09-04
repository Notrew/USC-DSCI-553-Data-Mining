import sys
from blackbox import BlackBox
import random
import binascii
import csv
import time

# bloom filterfing
# task1.py <input_file_path> <stream_size> <num_of_asks> <output_file_path>
input_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_path = sys.argv[4]
# input_path = "../data/input/users.txt"
# stream_size = 100
# num_of_asks = 30
# output_path = "../data/output/task1.csv"

s_time = time.time()
#  keep a global filter bit array and the length is 69997.
bit_array = [0]*69997
global_users = set()

def generate_hash_funcs(n_hash):
    res = []
    a = random.sample(range(1,round(time.time()/10000)),n_hash)
    b = random.sample(range(1,round(time.time()/10000)),n_hash)
    p = 9965
    for i in range(n_hash):
        def hashFunc(uid_int):
            hash_value  = ((a[i]*uid_int+b[i])%p)%69997
            return hash_value
        res.append(hashFunc)
    return res

def myhashs(uid_str,hash_funcs):
    uid_int = int(binascii.hexlify(uid_str.encode('utf8')), 16)
    res = []
    for func in hash_funcs:
        res.append(func(uid_int))
    return res

def cal_fpr(user_stream):
    false_positive = 0
    res = [0]*stream_size
    for i in range(len(user_stream)):
        tmp = []
        uid_hash = myhashs(user_stream[i],hash_funcs)
        for j in uid_hash:
            if bit_array[j] != 0:
                tmp.append(1)
        if len(tmp) == len(uid_hash):
            res[i] = 1
        if user_stream[i] not in global_users and res[i] == 1:
            false_positive += 1
    false_nagetive = stream_size-sum(res)
    fpr = false_positive/(false_positive+false_nagetive)
    return fpr

bx = BlackBox()
res = []
n_hash = 10
for i in range(num_of_asks):
    user_stream = bx.ask(input_path,stream_size)
    hash_funcs = generate_hash_funcs(n_hash)
    fpr = cal_fpr(user_stream)
    res.append((i,fpr))
    for uid in user_stream:
        global_users.add(uid)
        for i in myhashs(uid,hash_funcs):
            bit_array[i] = 1
            
# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to txt, format:
with open(output_path,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["Time","FPR"])
    for i in res:
        writer.writerow(i)

#export PYSPARK_PYTHON=python3.6                                                                                  
#export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64  
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit 
# --executor-memory 4G --driver-memory 4G 
# task1.py 2 "../resource/asnlib/publicdata/users.txt" "./task1.csv"