import time
import numpy as np



# for k in range(12):
#     start_time = time.time()
#     bf = []
#     vt = []
#     pt = []
#     gf = []
#     for i in range(12):
#         data = np.load(f'./catData/data_{i}.npz')
#         print(data['pt'].shape)
#         num = data['bf'].shape[0]//12
#         if k != 11:
#             if i == 0:
#                 bf = data['bf'][num*k: num*(k+1)]
#                 vt = data['vt'][num*k: num*(k+1)]
#                 pt = data['pt'][num*k: num*(k+1)]
#                 gf = data['gf'][num*k: num*(k+1)]
#             else:
#                 bf = np.vstack([bf, data['bf'][num*k: num*(k+1)]])
#                 vt = np.vstack([vt, data['vt'][num*k: num*(k+1)]])
#                 pt = np.vstack([pt, data['pt'][num*k: num*(k+1)]])
#                 gf = np.vstack([gf, data['gf'][num*k: num*(k+1)]])
#         else:
#             if i == 0:
#                 bf = data['bf'][num*11: -1]
#                 vt = data['vt'][num*11: -1]
#                 pt = data['pt'][num*11: -1]
#                 gf = data['gf'][num*11: -1]
#             else:
#                 bf = np.vstack([bf, data['bf'][num*11: -1]])
#                 vt = np.vstack([vt, data['vt'][num*11: -1]])
#                 pt = np.vstack([pt, data['pt'][num*11: -1]])
#                 gf = np.vstack([gf, data['gf'][num*11: -1]])
#         print()
#         print(i)
#         print(num)
#         print(vt.shape)
#         print()
#     print()
#     assert bf.shape != vt.shape or pt.shape != vt.shape or gf.shape != vt.shape
#     print(bf.shape)
#     print(vt.shape)
#     print(pt.shape)
#     print(gf.shape)
#     np.savez_compressed(f'./data/data_{k}.npz', bf=bf, vt=vt, pt=pt, gf=gf)
#     end_time = time.time()
#     print("Time:", end_time - start_time)
#     print()
#

data = np.load('./split/data_0_0.npz')
print(data['bf'].shape)
