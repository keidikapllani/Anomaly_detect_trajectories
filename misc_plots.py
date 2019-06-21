# -*- coding: utf-8 -*-
"""
Plot denoising graphs real data.

@author: kk2314
"""

import numpy as np
from utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

#%%
'''
Load Data
'''
cntrl = 0
data_smooth_1, data_raw_1 = load_data(cntrl,True)
cntrl = 1
data_smooth_2,data_raw_2 = load_data(cntrl,True)
cntrl = 2
data_smooth_3, data_raw_3 = load_data(cntrl,True)


dwt_vel_1 = sio.loadmat('dwt_vel_1')['v_de']
dwt_vel_2 = sio.loadmat('dwt_vel_2')['v_de']
dwt_vel_3 = sio.loadmat('dwt_vel_3')['v_de']
data_smooth_1[:,7] = dwt_vel_1[:,0]
data_smooth_2[:,7] = dwt_vel_2[:,0]
data_smooth_3[:,7] = dwt_vel_3[:,0]


data_smooth_all = np.vstack((data_smooth_1,data_smooth_2,data_smooth_3))

#%%
'''
Plotting distribution of velocities over the entire dataset
'''
plt.figure(dpi=150)
ax = sns.distplot(data_smooth_all[:,4],kde=True)
plt.xlim(0, 75)
plt.xlabel('Velocity (km/h)')
plt.ylabel('Prob. distribution')
plt.tight_layout()
#plt.close()

#%%
'''
Plotting speed for signle vehicle from raw, sema and dwt denosing

'''
indx_car = np.where(data_smooth_1[:,0]==400)[0]
car = data_smooth_1[indx_car,:]
car_raw = data_raw_1[indx_car,:]

fig, ax = plt.subplots(2, 2,gridspec_kw={'hspace': 0.2, 'wspace': 0})
ax[0,0].plot(car_raw[1:-1,1],car_raw[1:-1,4],'g', linewidth=1.5,label='NGSIM')
ax[0,0].plot(car[1:-1,1],car[1:-1,4],'b', linewidth=1.5,label='sEMA filtering')
ax[0,0].plot(car[1:-1,1],car[1:-1,7],'r', linewidth=1.5,label='DWT Denoised')
ax[0,0].set_title('(a) Velocity data for vehicle 400 in dataset 1')
ax[0,0].legend()

indx_car = np.where(data_smooth_2[:,0]==1458)[0]
car = data_smooth_2[indx_car,:]
car_raw = data_raw_2[indx_car,:]
ax[0,1].plot(car_raw[1:-1,1],car_raw[1:-1,4],'g', linewidth=1.5,label='NGSIM')
ax[0,1].plot(car[1:-1,1],car[1:-1,4],'b', linewidth=1.5,label='sEMA filtering')
ax[0,1].plot(car[1:-1,1],car[1:-1,7],'r', linewidth=1.5,label='DWT Denoised')
ax[0,1].set_title('(b) Velocity data for vehicle 1458 in dataset 2')
ax[0,1].legend()

indx_car = np.where(data_smooth_3[:,0]==245)[0]
car = data_smooth_3[indx_car,:]
car_raw = data_raw_3[indx_car,:]
ax[1,0].plot(car_raw[1:-1,1],car_raw[1:-1,4],'g', linewidth=1.5,label='NGSIM')
ax[1,0].plot(car[1:-1,1],car[1:-1,4],'b', linewidth=1.5,label='sEMA filtering')
ax[1,0].plot(car[1:-1,1],car[1:-1,7],'r', linewidth=1.5,label='DWT Denoised')
ax[1,0].set_title('(c) Velocity data for vehicle 245 in dataset 3')
ax[1,0].legend()

indx_car = np.where(data_smooth_3[:,0]==856)[0]
car = data_smooth_3[indx_car,:]
car_raw = data_raw_3[indx_car,:]
ax[1,1].plot(car_raw[1:-1,1],car_raw[1:-1,4],'g', linewidth=1.5,label='NGSIM')
ax[1,1].plot(car[1:-1,1],car[1:-1,4],'b', linewidth=1.5,label='sEMA filtering')
ax[1,1].plot(car[1:-1,1],car[1:-1,7],'r', linewidth=1.5,label='DWT Denoised')
ax[1,1].set_title('(d) Velocity data for vehicle 856 in dataset 3')
ax[1,1].legend()

for ax in ax.flat:
    ax.set(ylabel='Velocity (km/h)', xlabel='Time (h)')
    ax.label_outer()
fig.set_dpi(150)


#%%

'''
Plotting distribution of velocities over the entire dataset for raw denoised and smoothed
'''
plt.figure(dpi=150)
ax = sns.kdeplot(data_raw_1[:,4],bw=.5,label='NGSIM',color="g")
ax = sns.kdeplot(data_smooth_1[:,7],bw=.5,label='DWT Denoised')
ax = sns.kdeplot(data_smooth_1[:,4],bw=.5,label='sEMA Smoothed',color="r")
#sns.plt.show()
plt.xlim(0, 75)
plt.xlabel('Velocity (km/h)')
plt.ylabel('Prob. distribution')
plt.legend()
plt.tight_layout()
plt.savefig('Figures/denoising/Dist_den_semma_1.png')

plt.figure(dpi=150)
ax = sns.kdeplot(data_raw_2[:,4],bw=.5,label='NGSIM',color="g")
ax = sns.kdeplot(data_smooth_2[:,7],bw=.5,label='DWT Denoised')
ax = sns.kdeplot(data_smooth_2[:,4],bw=.5,label='sEMA Smoothed',color="r")
#sns.plt.show()
plt.xlim(0, 75)
plt.xlabel('Velocity (km/h)')
plt.ylabel('Prob. distribution')
plt.legend()
plt.tight_layout()
plt.savefig('Figures/denoising/Dist_den_semma_2.png')

plt.figure(dpi=150)
ax = sns.kdeplot(data_raw_3[:,4],bw=.5,label='NGSIM',color="g")
ax = sns.kdeplot(data_smooth_3[:,7],bw=.5,label='DWT Denoised')
ax = sns.kdeplot(data_smooth_3[:,4],bw=.5,label='sEMA Smoothed',color="r")
#sns.plt.show()
plt.xlim(0, 75)
plt.xlabel('Velocity (km/h)')
plt.ylabel('Prob. distribution')
plt.legend()
plt.tight_layout()
plt.savefig('Figures/denoising/Dist_den_semma_3.png')



#%%


