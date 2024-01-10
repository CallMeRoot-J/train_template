import time
import numpy as np


def merge_data(num_merge_files, ):
    index_f = np.load('./index.npz')
    index = index_f['index']
    for k in range(num_merge_files):
        start_time = time.time()
        bf = []
        vt = []
        pt = []
        gf = []
        for i in range(k*400, (k+1)*400):
            data = np.load(f'./data/data_{index[i]}.npz')
            print(data['vt'].shape)
            if i == k*400:
                bf = data['bf']
                vt = data['vt']
                pt = data['pt']
                gf = data['gf']
            else:
                bf = np.vstack([bf, data['bf']])
                vt = np.vstack([vt, data['vt']])
                pt = np.vstack([pt, data['pt']])
                gf = np.vstack([gf, data['gf']])
            print()
            print(i)
            print(vt.shape)
            print()
        print()
        assert bf.shape != vt.shape or pt.shape != vt.shape or gf.shape != vt.shape
        print(bf.shape)
        print(vt.shape)
        print(pt.shape)
        print(gf.shape)
        np.savez_compressed(
            f'./catData/data_{k}.npz', bf=bf, vt=vt, pt=pt, gf=gf)
        end_time = time.time()
        print("Time:", end_time - start_time)
        print()

    start_time = time.time()
    bf = []
    vt = []
    pt = []
    gf = []
    for i in range(11*400, 4777):
        data = np.load(f'./data/data_{index[i]}.npz')
        print(data['vt'].shape)
        if i == 4400:
            bf = data['bf']
            vt = data['vt']
            pt = data['pt']
            gf = data['gf']
        else:
            bf = np.vstack([bf, data['bf']])
            vt = np.vstack([vt, data['vt']])
            pt = np.vstack([pt, data['pt']])
            gf = np.vstack([gf, data['gf']])
        print()
        print(i)
        print(vt.shape)
        print()
    print()
    assert bf.shape != vt.shape or pt.shape != vt.shape or gf.shape != vt.shape
    print(bf.shape)
    print(vt.shape)
    print(pt.shape)
    print(gf.shape)
    np.savez_compressed(f'./catData/data_{11}.npz', bf=bf, vt=vt, pt=pt, gf=gf)
    end_time = time.time()
    print("Time:", end_time - start_time)
    print()
