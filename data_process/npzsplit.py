import os
import numpy as np


file = os.listdir('./stage0Data')
for i in range(len(file)):
    data = np.load(f'./stage0Data/{file[i]}')
    rand_state = np.random.get_state()
    bf = data['bf']
    vt = data['vt']
    np.random.set_state(rand_state)
    np.random.shuffle(bf)
    np.random.shuffle(vt)
    num = bf.shape[0] // 50
    for j in range(49):
        np.savez_compressed(f'./split/data_{i}_{j}.npz', bf=bf[num*j: num*(j+1)], vt=vt[num*j: num*(j+1)])
    np.savez_compressed(f'./split/data_{i}_49.npz', bf=bf[num * 49:], vt=vt[num * 49:])



