import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
from plotly.graph_objs import Scatter3d

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

r_e = 6378e3
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r_e * np.outer(np.cos(u), np.sin(v))
y = r_e * np.outer(np.sin(u), np.sin(v))
z = r_e * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_wireframe(x, y, z)

ax.set_aspect("equal")
ax.legend()

fig = plotly.graph_objs.Figure()
fig.add_trace(
    Scatter3d(
        x=intercept_r.T[0],
        y=intercept_r.T[1],
        z=intercept_r.T[2],
        mode="lines",
        name="Transfer Orbit",
    )
)
fig.add_trace(
    Scatter3d(x=kkv_r.T[0], y=kkv_r.T[1], z=kkv_r.T[2], mode="lines", name="KKV")
)
fig.add_trace(
    Scatter3d(
        x=target_r.T[0], y=target_r.T[1], z=target_r.T[2], mode="lines", name="ICBM"
    )
)
fig.write_html("out.html")

plt.show()
