import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection="3d")

with open(sys.argv[1], "rb") as f:
    (
        intercept_t,
        intercept_r,
        kkv_t,
        kkv_r,
        target_t,
        target_r,
        burn_pos,
        intercept_pos,
    ) = pickle.load(f)

intercept_r = np.array(intercept_r)
kkv_r = np.array(kkv_r)
target_r = np.array(target_r)


ax.plot(*intercept_r.T, label="Intercept Orbit")
ax.plot(*kkv_r.T, label="KKV Orbit")
ax.plot(*target_r.T, label="Target Orbit")

ax.text(*burn_pos, "Transfer Burn")
ax.text(*intercept_pos, "KKV Impact")

ax.set_aspect("equal")
ax.legend()
plt.show()
