import sys
from blackbox import BlackBox
import random
import binascii
import csv
import time

#  Flajolet-Martin algorithm
# task2.py <input_file_path> <stream_size> <num_of_asks> <output_file_path>
input_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_path = sys.argv[4]
# input_path = "../data/input/users.txt"
# stream_size = 300
# num_of_asks = 30
# output_path = "../data/output/task2.csv"

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

def cal_estimation(user_stream):
    tmp = [0]*n_hash
    for i in range(len(user_stream)):
        uid_hash = myhashs(user_stream[i],hash_funcs)
        for j in range(len(uid_hash)):
            binary_num = bin(uid_hash[j])
            n_zero = len(binary_num.split("1")[-1])
            if n_zero >=tmp[j]:
                tmp[j] = n_zero
    estimation = round(sum([2**i for i in tmp])/n_hash)
    return estimation

s_time = time.time()
global_users = set()
bx = BlackBox()
res = []
n_hash = 10
for i in range(num_of_asks):
    user_stream = bx.ask(input_path,stream_size)
    hash_funcs = generate_hash_funcs(n_hash)
    estimation = cal_estimation(user_stream)
    res.append((i,stream_size,estimation))

# final result, 0.2 <= (sum of all your estimations / sum of all ground truths) <= 5
# estimation_sum = sum([i[2] for i in res])
# estimation_sum/(300*30)

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv, format:
with open(output_path,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["Time","Ground Truth","Estimation"])
    for i in res:
        writer.writerow(i)