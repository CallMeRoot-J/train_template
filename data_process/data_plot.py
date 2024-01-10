import numpy as np
import os
import matplotlib.pyplot as plt

smoth = 10
TRAIN_SMOTH = 100 * smoth
TEST_SMOTH = 1 * smoth * 5
draw_log_dir = [
    #
    "4b64f",
    "conv3NNUE",
    "dhb_conv3",
    "splitedConv5"
    # "splitedConv5x",
    # "conv3xNNUE",
]
for d in draw_log_dir:
    all_loss_files = os.listdir(f"../log/{d}")
    loss = []
    for i in range(len(all_loss_files)):
        data = np.load(f"../log/{d}/" + all_loss_files[i])
        a = np.array(data["train"]).reshape((1, len(data["train"])))
        if i == 0:
            loss = a[0]
        else:
            loss = np.hstack([loss, a[0]])
    mean_loss = []
    n = len(loss) // TRAIN_SMOTH
    for i in range(n):
        l1 = loss[i * TRAIN_SMOTH : (i + 1) * TRAIN_SMOTH]
        mean_loss.append(np.mean(l1))
    x = list(range(n))
    # xticks(pylab.np.linspace(0, 1, 50, endpoint=True))
    # yticks(pylab.np.linspace(0, 1, 100, endpoint=True))
    plt.plot(x, mean_loss, label=f"{d}")
    plt.plot(x[-1], mean_loss[-1], "gs")
    plt.annotate(
        f"({mean_loss[-1]:<2.4f})",
        xytext=(x[-1], mean_loss[-1]),
        xy=(x[-1], mean_loss[-1]),
    )
plt.grid(True)  # 网格线
plt.xlabel(f"{TRAIN_SMOTH} step")
plt.ylabel("loss")
plt.legend()
plt.show()
