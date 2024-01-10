import os
import numpy as np

file = os.listdir('./data')
for i in range(len(file)):
    data = np.load(f'./data/{file[i]}')
    gf = data['gf'][:, 0]
    bf = data['bf'][gf == 0]
    bf = bf[:, 0: 2, :, :]
    vt = data['vt'][gf == 0]
    print(vt.shape)
    print(bf.shape)
    np.savez_compressed(f'./stage0Data/data_{i}.npz', bf=bf, vt=vt)

