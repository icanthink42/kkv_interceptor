import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection="3d")

with open(sys.argv[1], "rb") as f:
    t, r = pickle.load(f)

r = np.array(r)

ax.plot(*r.T)
ax.set_aspect("equal")
plt.show()
