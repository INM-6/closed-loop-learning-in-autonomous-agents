import numpy as np
import pylab as plt
import json

pos = []

for i, x in enumerate(np.linspace(-1.0, 1.0, 5)):
    for j, y in enumerate(np.linspace(-1.0, 1.0, 5)):
        pos.append([x, y, 0.2, 0.2])

with open("grid_pos.json", "w+") as f:
    json.dump(pos, f)


