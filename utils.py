#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Collection of functions used in the main.py
@author: Keidi Kapllani
"""

import numpy as np
import pywt
import pandas as pd
from wrcoef import wavedec1, wrcoef1
import urllib.request
import requests

#%%
def load_data(cntrl, force_recalc = False):
    print('LOG: Loading data ')
    '''
    Function to load data and perform smoothing and denoising. This function stores the processed data.
    If a saved array exist, it loads from that array. 
    INPUT: cntrl: Select which time-group to load (0,1,2)
           force_recalc: Forces recalculation from original dataset
           
    OUTPUT: data_smooth: Smooth and denoised data
    '''

    data = ['1','2', '3'] 
    #columns to keep for working data
    keep = [0,3,6,10,11,13,16]
    
    try:
        if force_recalc:
            print('Forcing smoothing for original data')
            raise UnboundLocalError('Forcing smoothing for original data')
            
        file_name = 'matricies/data_'+data[cntrl]+'_smoooth_de.npy'
        data_smooth = np.load(file_name)
        file_name = 'matricies/data_'+data[cntrl]+'_raw.npy'
        data_raw = np.load(file_name)
        print('LOG: Data loaded from pre-processed arrays ')
        
    except:
        print('LOG: Peforming smoothing & denoising on data ')
        if cntrl == 0:
            try:
                data_first  = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
            except:
                print('Dowloading data for 0750am-0805am ')
                url = 'https://drive.google.com/open?id=1FKC3TrKFDAsK0gQO_YRujpal6gPQtJiB'  
                download_file_from_google_drive(url,'data/trajectories-0750am-0805am.txt')
                data_first  = pd.read_csv('data/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
            timestamp = pd.to_datetime(data_first[3],unit='ms')
            data_first[3] = time_fix(timestamp)
            data_raw = data_first.values 
            data_raw = data_raw[:,keep]
            data_raw[:,2] = data_raw[:,2]* 0.0003048  #convert position data to km
            data_raw[:,6] = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4] = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth= smooth_data(data_ext)
            np.save( 'matricies/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matricies/data_'+data[cntrl]+'_raw',data_raw)
            
        #Second time-group    
        elif cntrl==1:
            data_second = pd.read_csv('data/0805am-0820am/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
            timestamp = pd.to_datetime(data_second[3],unit='ms')
            data_second[3] = time_fix(timestamp)
            data_raw = data_second.values 
            data_raw = data_raw[:,keep]
            data_raw[:,2] = data_raw[:,2]* 0.0003048  #convert to km
            data_raw[:,6] = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4] = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth = smooth_data(data_ext)
            np.save( 'matricies/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matricies/data_'+data[cntrl]+'_raw',data_raw)
        #Third time-group    
        elif cntrl==2:
            data_third  = pd.read_csv('data/0820am-0835am/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
            timestamp = pd.to_datetime(data_third[3],unit='ms')
            data_third[3] = time_fix(timestamp)
            data_raw = data_third.values 
            data_raw = data_raw[:,keep]
            data_raw[:,2] = data_raw[:,2]* 0.0003048  #convert to km
            data_raw[:,6] = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4] = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth = smooth_data(data_ext)
            np.save( 'matricies/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matricies/data_'+data[cntrl]+'_raw',data_raw)
        
        else:
            raise UnboundLocalError('Please select either dataset 1,2 or 3')
        
            
    
    print('LOG: Done')
    return data_smooth, data_raw





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
    coef, _t=pywt.cwt(x,a,'mexh')
    coef = coef.clip(min=0)
    E = (1/64) * np.sum(np.square(np.abs(coef)), axis=0)
    return E

#%%
    'Performs smoothing and wavelet denoising on data'
def smooth_data(data):
    print('Smoothing and Denoising')
    car_id_list = np.unique(data[:,0]).astype(int)
   
    dt = 0.1/3600; # data point time interval
    Tx = 0.5/3600; # smoothing width (position data)
    Tv = 1/3600; # smoothing width (velocity data)   
    
    for i in range(0,car_id_list.size):
        idx = np.where(data[:,0]==car_id_list[i])[0]
        pos = data[idx,2]
        pos1 = sema_smoothing(pos,dt,Tx)
        vel_dif = diff_v_a(pos1,dt)
        vel_smooth = sema_smoothing(vel_dif,dt,Tv)
        dwt_vel = denoise(data[idx,4])
        vel = np.insert( vel_smooth, 0, 0)
        vel = np.append( vel, 0)
        data[idx,2] = pos1
        data[idx,4] = vel
        data[idx,7] = dwt_vel
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
    lvl = 4
    db4 = pywt.Wavelet('db4')
#    coeffs=pywt.wavedec(data, db4, level=lvl)
    C, L = wavedec1(data, wavelet=db4, level=lvl)
#    n = data.size
#    cA6cD_approx = pywt.waverec(coeffs, 'db4',axis = 0)
    D = wrcoef1(C, L, wavelet=db4, level=4)
#    (cA, cD) = pywt.dwt(data, 'db4')
#    smotheed = pywt.waverec()
    return D

#%%
def lane_changer(lane_data):
    """
    identify lane changers and return car ids of lane changers and non-lane changers
    INPUT: lane_data: Full data in the forma nx7
           
    OUTPUT: lane_normal: Car ids of non-lane changers
            lane_changers: Car ids of lane changers
    """
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
    """
    Classify peaks into Acceleration (A), Decceleration (D) or Steade-state (S)
    INPUT: locs_p: location of peaks in positive wave
           locs_n: locaiton of peaks in negative wave
           v:      velocity data
    OUTPUT: locs_sort: Index of peaks classified
            info_sort: Peak information; 0 - Decceleration (D) , 1-Acceleration (A), 3-Steade-state (S)
    """
    locs_pn = np.append(locs_p,locs_n)
    info = np.append(np.ones(locs_p.size),np.zeros(locs_n.size))
    locs_sort = np.sort(locs_pn)
    info_sort = info[np.argsort(locs_pn)]
    if locs_sort.size>0:
        for i in range(0,locs_sort.size-1):
            if(abs(v[locs_sort[i]]-v[locs_sort[i+1]]))<5:
                info_sort[i] = 2
        
        if locs_sort[locs_sort.size-1] > v.size-64:
            locs_sort = locs_sort[:-1]
            info_sort = info_sort[:-1]
#        if locs_sort[0] < 26:
#            locs_sort = locs_sort[1:]
#            info_sort = info_sort[1:]
#    
        
    return locs_sort, info_sort
#%%%
def time_fix(timestamp):
    """
    Convert pandas timestamp into hours represenation
    INPUT: timestamp: pandas timestamp data
    OUTPUT: t_fixed: Time data in hours
    """
    hour = timestamp.dt.hour.values - 7
    minutes = timestamp.dt.minute.values
    seconds = timestamp.dt.second.values
    ms = timestamp.dt.microsecond.values/1e6
    t_fixed = hour + np.round((((seconds+ms)/60)+minutes)/60,4)
    
    return t_fixed

#%%
def clust_assign(t_new, x_new, cid_new, t, x, cid):
    
    time_flag = False;
    id_flag = False;
    space_flag = False;
    
    'Check if the time difference is between 0.9 - 1 sec'
    time_diff = t_new - t
    if (time_diff  > -5/3600 and time_diff < 10/3600):
        time_flag = True
        
    'check if ther are different cars'
    if cid_new > cid:
        id_flag = True
    'Check if the space is more then 50m'
    if x_new - x < 0.5:
        space_flag = True
    
    
    if (time_flag and id_flag and space_flag):
        cluster_flag = 1
    else:
        cluster_flag = 0
    
    xt_dist = np.linalg.norm([abs(t-t_new),abs(x-x_new)])
    
    
    
    return cluster_flag, xt_dist

#%%%
    

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    
    
    
    
    
    
    
    
    