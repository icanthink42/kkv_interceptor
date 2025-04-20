import json
import sys

import matplotlib.pyplot as plt

t, r = json.loads(sys.argv[0])
plt.plot(t, *r)
plt.show()
