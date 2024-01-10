import numpy as np
import time
import random
import os

index = np.array([i for i in range(6)])
np.random.shuffle(index)
file = os.listdir("../data")
start_time = time.time()
for k in range(2):
    bf = []
    vt = []
    for i in range(3):
        data = np.load(f"../data/{file[index[i+k*3]]}")
        print(data["bf"].shape)
        if i == 0:
            bf = data["bf"]
            vt = data["vt"]
        else:
            bf = np.vstack([bf, data["bf"]])
            vt = np.vstack([vt, data["vt"]])
        print()
        print(i)
        print(vt.shape)
        print()
    print()
    print(bf.shape)
    print(vt.shape)
    np.savez_compressed(f"../d/data_{k}.npz", bf=bf, vt=vt)
    end_time = time.time()
    print("Time:", end_time - start_time)
    print()
