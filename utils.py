#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Collection of functions used in the main.py
@author: Keidi Kapllani
"""

import numpy as np
import pywt
from scipy import signal


#%%
def diff_v_a(x_in,dt):
# Perform numerical derivative to estimate instantenous velocity from poistion data. 
    
    N = np.shape(x_in)
    vel = np.zeros(N[0])
    for i in range(1,N[0]-1):
        vel[i-1] = (x_in[i+1]-x_in[i-1])/(2*dt)
        
    vel = vel[:N[0]-2]    
    return vel

#%%

def sema_smoothing(x_in,dt,T):
    delta = T/dt
    N = np.shape(x_in)
    x_out = np.zeros(N[0])
    for i in range(1,N[0]+1):
        D = int(min(3*delta,i-1,N[0]-i))
        k = np.arange(i-D,i+D+1)
        p = -np.abs(i-k)/delta
        exx = np.exp(p)
        Z = np.sum(exx)
        xa = x_in[k-1]
        x_out[i-1] = (1/Z) * np.sum(xa.T*exx)
    return x_out

#%%
' Calculate the wavelet energy for a signal x'    
def wvlt_ener(x):
    a = np.arange(1,64.1,0.1)
    coef=signal.cwt(x,signal.ricker,a)
    coef = coef.clip(min=0)
    E = (1/64) * np.sum(np.square(np.abs(coef)), axis=0)
    return E

#%%
    'Performs smoothing and wavelet denoising on data'
def smooth_data(data):
    car_id_list = np.unique(data[:,0]).astype(int)
   
    dt = 0.1/3600; # data point time interval
    Tx = 0.5/3600; # smoothing width (position data)
    Tv = 1/3600; # smoothing width (velocity data)   
    
    for i in range(0,car_id_list.size):
        idx = np.where(data[:,0]==car_id_list[i])
        idx = idx[0]
        pos = data[idx,2]
        vel_dif = diff_v_a(pos,dt)
        pos1 = sema_smoothing(pos,dt,Tx)
        vel_smooth = sema_smoothing(vel_dif,dt,Tv)
        vel_final = denoise(vel_smooth)
        vel = np.insert( vel_final, 0, 0)
        vel = np.append( vel, 0)
        data[idx,2] = pos1
        data[idx,4] = vel
    return data
#%%

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))

#%%
def denoise(data):
    lvl = 5
    db4 = pywt.Wavelet('db4')
    cA5,cD5, cD4, cD3, cD2, cD1=pywt.wavedec(data, db4, level=lvl)
    n = data.size
    cA6cD_approx = pywt.upcoef('a',cA5,'db4',take=n, level=lvl)
    return cA6cD_approx

#%%
def lane_changer(lane_data):
    lane_changers = []
    lane_normal = []
    car_id_list = np.unique(lane_data[:,0]).astype(int)
    for i in range(0,car_id_list.size):
        car = lane_data[np.where(lane_data[:,0]==car_id_list[i])]
        lanes = np.unique(car[:,5])
        if lanes.size == 1:
            lane_normal.append(car_id_list[i])
        else:
            lane_changers.append(car_id_list[i])
        
    return lane_normal, lane_changers    

#%%%
def peak_classify(locs_p, locs_n, v):
    
    locs_pn = np.append(locs_p,locs_n)
    info = np.append(np.ones(locs_p.size),np.zeros(locs_n.size))
    locs_sort = np.sort(locs_pn)
    info_sort = info[np.argsort(locs_pn)]
    if info.size>0:
#        for i in range(0,locs_sort.size-1):
#            if abs(v[locs_sort[i]]-v[locs_sort[i+1]])<5:
#                info_sort[i] = 3
        
        if locs_sort[info.size-1] > v.size-64:
            locs_sort = locs_sort[:-1]
            info = info[:-1]
            
    
        
    return locs_sort, info_sort
