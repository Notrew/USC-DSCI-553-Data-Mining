import sys
from blackbox import BlackBox
import random
import csv
import time

#  Fixed Size Sampling
# task3.py <input_file_path> <stream_size> <num_of_asks> <output_file_path>
input_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_path = sys.argv[4]
# input_path = "../data/input/users.txt"
# stream_size = 100
# num_of_asks = 30
# output_path = "../data/output/task3.csv"

s_time = time.time()
bx = BlackBox()
random.seed(553)
tmp = []
res = []
cnt = 0

# reservoir_sampling
for i in range(num_of_asks):
    user_stream = bx.ask(input_path,stream_size)
    if i == 0:
        tmp = tmp+user_stream
        cnt += stream_size
    else:
        for user in user_stream:
            cnt += 1
            prob = random.random()
            if prob<100/cnt:
                index = random.randint(0,99)
                tmp[index] = user
    res.append((cnt,tmp[0],tmp[20],tmp[40],tmp[60],tmp[80]))

# less than 100 second
e_time = time.time()
duration = e_time-s_time
print("Duration:",duration)

# write to csv format:
with open(output_path,"w") as f:
    writer = csv.writer(f)
    writer.writerow(["seqnum","0_id","20_id","40_id","60_id","80_id"])
    for i in res:
        writer.writerow(i)
