from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import pywt
from utils import *
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy import signal

#%%
# Load data and peform smoothing denoising
#
#
 
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

data_first  = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
#data_second = pd.read_csv('data/0805am-0820am/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
#data_third  = pd.read_csv('data/0820am-0835am/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
data_first.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
#data_second.columns = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
#data_third.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']



data_raw = data_first.values  #+ data_second.values + data_third.values
#values to keep
keep = [0,3,6,10,11,13,16]
data_raw = data_raw[:,keep]

data_raw[:,1] = (data_raw[:,1]-data_raw[0,1])/1000 #fix time to ms since start of observation
data_raw[:,2] = data_raw[:,2]* 0.0003048 #convert to km

#smooth and denoise the data
data_smooth = smooth_data(data_raw)
#del data_raw
del data_first
#load presmoothed data
smooth_data1 = sio.loadmat('smooth_de_data_1')
v_de = smooth_data1['v_de']
x_data_smooth = smooth_data1['X_data_smooth']
#time_data = sio.loadmat['t_data']
#t_data = t_data['t_data']
data_smooth[:,4] = v_de[:,0]
data_smooth[:,2] = x_data_smooth[:,0]
data_smooth[:,1] = t_data[:,0]
del v_de
del x_data_smooth
del smooth_data1
#%%
#database = sio.loadmat('database.mat')
#lane_ID = sio.loadmat('lane_ID')
#
#data = database['database']
#
#lane_1_id = lane_ID['lane1_ID']
#data[:,1] = (data[:,1]-1118840000000-8770700)/1000

index_lane1 = np.where(data_smooth[:,5]==1)

lane_1 = data_raw[index_lane1[0],:]
lane_1_id = np.unique(lane_1[:,0])
# select one car for testing
idx_car = np.where(data_smooth[:,0]==762)
car = data_smooth[idx_car[0],:]


for i in range(0,lane_1_id.size):
    car_id = lane_1_id[i]
    
    indexes = np.where(lane_1[:,0]==car_id)
    car = lane_1[indexes[0],:]
    
    plt.plot(car[:,1],car[:,2],'b')
    
    
plt.show()

coef, freqs=pywt.cwt(car[:,2],car[:,1],'mexh')
plt.matshow(coef) # doctest: +SKIP
plt.show()

#%%
'rough paper'
'trying to implement fwt_single'
#find data in time windows
data_smooth_time_wind = data_smooth[(data_smooth[:,1]>180) & (data_smooth[:,1]<360.1)]
#timed_test = data_smooth[np.where(data_smooth[:,1]>180)]
#timed_test = timed_test[np.where(timed_test[:,1]<360.1)]
#find lane changers
lane_nor_id, lane_chang_id = lane_changer(data_smooth)
#find ids in lane 1

idx_lane1 = np.where(data_smooth_time_wind[:,5]==1)
lane_1 = data_smooth_time_wind[idx_lane1[0],:]
ids = np.unique(lane_1[:,0]).astype(int)
#find non-lane changers in lane 1 
lane_1_normal_ids = np.intersect1d(lane_nor_id,ids)

time_wind = sio.loadmat('time_window_mat.mat')
time_window_mat = time_wind['time_window_car_ID'][0,:]
lane_1_normal_ids = np.intersect1d(lane_nor_id,time_window_mat)
lane_1_changers_ids = np.intersect1d(lane_chang_id,time_window_mat)

#%%
for i in range(0,lane_1_normal_ids.size):
    #get a single vehilce
    car = data_smooth[np.where(data_smooth[:,0]==lane_1_normal_ids[i])]
    if car[:,4].size <300:
        continue
#    print(car[:,4].size)
    vel = car[2:-2,4]
    v_out = vel[64:-64]
    pos = car[:,2]
    t = car[:,1]
    #extend by 240 on both sides
    zr = np.ones(240,)*vel[vel.size-1]
    vel = np.append(vel,zr)
    zr = np.ones(240,)*vel[0]
    vel = np.insert( vel, 0, zr)
    vel_rev = np.max(vel) - vel
    #calculate energy
    e_p = wvlt_ener(vel)
    e_n = wvlt_ener(vel_rev)
    #remove extended data
    e_p = e_p[304:-304]
    e_n = e_n[304:-304]
    pos = pos[66:-66]
    t = t[66:-66]
    thresh = 0.3e4
    locs_p, _d = signal.find_peaks(e_p,height=thresh)
    locs_n, _d = signal.find_peaks(e_n,height=thresh)
    
    locs, info = peak_classify(locs_p, locs_n,vel)
    
    plt.plot(t,pos,'k', linewidth=0.8)
    
    for j in range(0,info.size):
        if info[j] == 1:
            plt.scatter(t[locs[j]],pos[locs[j]],marker='x',c='blue')
        if info[j] == 0:
            plt.scatter(t[locs[j]],pos[locs[j]],marker='o',c='red')
        else:
            continue

for i in range(0,lane_1_changers_ids.size):
    car = data_smooth[np.where(data_smooth[:,0]==lane_1_changers_ids[i])]
    pos = car[2:-2,2]
    t = car[2:-2,1]
    plt.plot(t,pos,'k', linewidth=0.8,ls='--')
    
plt.show()


























