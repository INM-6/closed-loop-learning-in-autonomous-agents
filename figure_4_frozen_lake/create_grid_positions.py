import numpy as np
import json

n = 16 

pos = []

for i, x in enumerate(np.linspace(-1., 1., n)):
   pos.append([x, 0.01])

with open('grid_pos.json', 'w+') as f:
    json.dump(pos, f)
