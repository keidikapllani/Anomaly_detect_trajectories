# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:10:58 2019

@author: kk2314
"""
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from utils import load_data, lane_changer, wvlt_ener, peak_classify
import matplotlib.pyplot as plt

#%%

car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[100])]
vel = car[2:-2,4]
pos = car[2:-2,2]
t = car[2:-2,1]
points = np.array([t, pos]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(1, 1, figsize=(15,10))

norm = plt.Normalize(vel.min(), vel.max())
lc = LineCollection(segments, cmap='inferno', norm=norm)
# Set the values used for colormapping
lc.set_array(vel)
lc.set_linewidth(2)
line = axs.add_collection(lc)
fig.colorbar(line, ax=axs)
axs.set_xlim(t.min(), t.max())
axs.set_ylim(pos.min(), pos.max())
plt.show()

plt.show()


plt.plot(vel,pos)
