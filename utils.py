#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Collection of functions used in the ofr the final year projects.

@author: Keidi Kapllani
"""

import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import signal
from scipy import stats 
from wrcoef import wavedec1, wrcoef1
#!pip install gdown
import gdown
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
            
        file_name   = 'matrices/data_'+data[cntrl]+'_smoooth_de.npy'
        data_smooth = np.load(file_name)
        file_name   = 'matrices/data_'+data[cntrl]+'_raw.npy'
        data_raw    = np.load(file_name)
        print('LOG: Data loaded from pre-processed arrays ')
        
    except:
        print('LOG: Peforming smoothing & denoising on data ')
        if cntrl == 0:
            try:
                data_first  = pd.read_csv('data/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
            except:
                print('LOG: Downloading data for 0750am-0805am ')
                url    = 'https://drive.google.com/uc?id=1E7_ABEvLcWFUC6xkknJSreKNAU12v-Fk'
                output = 'data/trajectories-0750am-0805am.txt'
                gdown.download(url, output, quiet=False)
                data_first  = pd.read_csv('data/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
                
            timestamp       = pd.to_datetime(data_first[3],unit='ms')
            data_first[3]   = time_fix(timestamp)
            data_raw        = data_first.values 
            data_raw        = data_raw[:,keep]
            data_raw[:,2]   = data_raw[:,2]* 0.0003048  #convert position data to km
            data_raw[:,6]   = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4]   = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext        = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth     = smooth_data(data_ext)
            np.save( 'matrices/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matrices/data_'+data[cntrl]+'_raw',data_raw)
            
        #Second time-group    
        elif cntrl==1:
            try:
                data_second = pd.read_csv('data/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
            except:
                print('LOG: Dowloading data for 0805am-0820am ')
                url    = 'https://drive.google.com/uc?id=1flg3VYnOOAQa3we74WPS8MsI0EuUdLqi'
                output = 'data/trajectories-0805am-0820am.txt'
                gdown.download(url, output, quiet=False)
                data_second = pd.read_csv('data/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
            timestamp       = pd.to_datetime(data_second[3],unit='ms')
            data_second[3]  = time_fix(timestamp)
            data_raw        = data_second.values 
            data_raw        = data_raw[:,keep]
            data_raw[:,2]   = data_raw[:,2]* 0.0003048  #convert to km
            data_raw[:,6]   = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4]   = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext        = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth = smooth_data(data_ext)
            np.save( 'matrices/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matrices/data_'+data[cntrl]+'_raw',data_raw)
        #Third time-group    
        elif cntrl==2:
            try:
                data_third  = pd.read_csv('data/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
            except:
                print('LOG: Dowloading data for 0820am-0835am ')
                url    = 'https://drive.google.com/uc?id=1FKC3TrKFDAsK0gQO_YRujpal6gPQtJiB'
                output = 'data/trajectories-0820am-0835am.txt'
                gdown.download(url, output, quiet=False)
                data_third  = pd.read_csv('data/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
            timestamp = pd.to_datetime(data_third[3],unit='ms')
            data_third[3]   = time_fix(timestamp)
            data_raw        = data_third.values 
            data_raw        = data_raw[:,keep]
            data_raw[:,2]   = data_raw[:,2]* 0.0003048  #convert to km
            data_raw[:,6]   = data_raw[:,6]* 0.0003048  #convert space data to km
            data_raw[:,4]   = data_raw[:,4]* 1.09728  #convert space data to km
            data_ext        = np.zeros((data_raw.shape[0],8))
            data_ext[:,:-1] = data_raw
            data_smooth     = smooth_data(data_ext)
            np.save( 'matrices/data_'+data[cntrl]+'_smoooth_de',data_smooth)
            np.save( 'matrices/data_'+data[cntrl]+'_raw',data_raw)
        
        else:
            raise UnboundLocalError('Please select either dataset 1,2 or 3')
        
            
    
    print('LOG: Done')
    return data_smooth, data_raw
#%%
def diff_v_a(x_in,dt):
    '''
    # Perform numerical derivative to estimate instantenous velocity from poistion data. 
    '''
    N   = np.shape(x_in)
    vel = np.zeros(N[0])
    for i in range(1,N[0]-1):
        vel[i-1] = (x_in[i+1]-x_in[i-1])/(2*dt)
        
    vel = vel[:N[0]-2]    
    return vel

#%%

def sema_smoothing(x_in,dt,T):
    delta = T/dt
    N     = np.shape(x_in)
    x_out = np.zeros(N[0])
    for i in range(1,N[0]+1):
        D   = int(min(3*delta,i-1,N[0]-i))
        k   = np.arange(i-D,i+D+1)
        p   = -np.abs(i-k)/delta
        exx = np.exp(p)
        Z   = np.sum(exx)
        xa  = x_in[k-1]
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
    
def smooth_data(data):
    print('Smoothing and Denoising')
    car_id_list = np.unique(data[:,0]).astype(int)
   
    dt = 0.1/3600 # data point time interval
    Tx = 0.5/3600 # smoothing width (position data)
    Tv = 1/3600 # smoothing width (velocity data)   
    
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
    locs_pn   = np.append(locs_p,locs_n)
    info      = np.append(np.ones(locs_p.size),np.zeros(locs_n.size))
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
    hour    = timestamp.dt.hour.values - 7
    minutes = timestamp.dt.minute.values
    seconds = timestamp.dt.second.values
    ms      = timestamp.dt.microsecond.values/1e6
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
    
#%%
def anomaly_detect(data_smooth, frame_length =3/60, plot = 'Normal', start_frame = 0, num_frames = 1, start_lane = 0, num_lanes=1 ):
    '''
     Function to detect anomalies for 1 frame at a time. Returns peaks found. 
     
     INPUT: data_smooth: smooth data matrix for 1 time-period
           frame_length: length of each frame to analyse
           start_frame: index of starting frame (0 is first frame)
           num_frames: number of frames (NOTE: Number fo frames should be proportional to frame length)
           start_lane: Starting lane to consider (0- Lane 1 ... 4- Lane 5)
           num_lanes : Number of lanes to analyse (MAX 5)
           v:      velocity data
     OUTPUT: peaks_groups: All peaks detected in the form [Car ID, Peak Time, Peak Position , Peak Type]
    '''
    
    lanes   = ['1','2','3','4','5']
    car_ids = np.unique(data_smooth[:,0]).astype(int)
    for l in range(start_frame, start_frame+num_frames):
        #find data in time windows
        window_start = data_smooth[0,1] + frame_length*l 
        window_end   = window_start + frame_length
        windowed_car_ids = []
        for i in range(len(car_ids)):
            car = data_smooth[np.where(data_smooth[:,0]==car_ids[i])]
            if (car[0,1] > window_start and car[0,1]<window_end):
                windowed_car_ids = np.append(windowed_car_ids,car[0,0])
            else:
                continue
        lane_nor_id, lane_chang_id = lane_changer(data_smooth)
        peaks_groups = []
        
        for j in range(start_lane, start_lane+num_lanes):    
            print('LOG: Working lane '+lanes[j]+' in frame '+str(l+1))
            #find ids in lane j+1 
            
            idx_lane = np.where(data_smooth[:,5]==j+1)
            lane     = data_smooth[idx_lane[0],:]
            ids_lane = np.unique(lane[:,0]).astype(int)
            #find non-lane changers in frame
            no_change_time_wind_ids = np.intersect1d(lane_nor_id,windowed_car_ids).astype(int)
            #find lane changers in frame 
            lane_cngr_time_wind_ids = np.intersect1d(lane_chang_id,windowed_car_ids).astype(int)
            #find non-lane changers in lane i+1 in frame
            windowed_lane_cf_ids    = np.intersect1d(no_change_time_wind_ids,ids_lane).astype(int)
            #find non-lane changers in lane i+1 in frame 
            windowed_lane_lc_ids    = np.intersect1d(lane_cngr_time_wind_ids,ids_lane).astype(int)
            car      = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[0])]
            ylim_min = car[66,2]  
            ylim_max = car[len(car)-67,2]
            
            if plot == 'Heatmap':
                fig, axs = plt.subplots(1,1,figsize=(15,10))
            elif plot == 'Normal':
                plt.figure(figsize=(15,10))
            else:
                None
                
            for i in range(0,windowed_lane_cf_ids.size):
                #get a single vehicle
                car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[i])]
                
                if car[:,4].size <150:
                    continue
                
                vel   = car[ 2:-2 ,4]
                v_out = car[66:-66,4]
                pos   = car[66:-66,2]
                t     = car[66:-66,1]
                #extend by 240 on both sides
                zr  = np.ones(240,)*vel[vel.size-1]
                vel = np.append(vel,zr)
                zr  = np.ones(240,)*vel[0]
                vel = np.insert( vel, 0, zr)
                #reversved signal
                vel_rev = np.max(vel) - vel
                #calculate energy
                e_p = wvlt_ener(vel)
                e_n = wvlt_ener(vel_rev)
                #remove extended data
                e_p = e_p[304:-304]
                e_n = e_n[304:-304]
                thresh     = 1e4
                locs_p, _d = signal.find_peaks(e_p,height=thresh)
                locs_n, _d = signal.find_peaks(e_n,height=thresh)
                locs, info = peak_classify(locs_p, locs_n,v_out)
                if i == 0:
                    peak_ids   = np.repeat(car[0,0],info.size)
                    peaks_time = t[locs]
                    peaks_pos  = pos[locs]
                    peaks_info = info
                else:
                    peak_ids   = np.append(peak_ids,np.repeat(car[0,0],info.size))
                    peaks_time = np.append(peaks_time,t[locs])
                    peaks_pos  = np.append(peaks_pos,pos[locs])
                    peaks_info = np.append(peaks_info,info)
                    
                if plot == 'Normal':
                    plt.plot(t,pos,'k', linewidth=1) # normal plot
                    for k in range(0,info.size):
                        if info[k] == 1:
                            dec = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                              label='Deceleration',s=60, 
                                              facecolors='none', 
                                              edgecolors='b')
                        if info[k] == 0:
                            acc = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                               s=60, 
                                              facecolors='none', 
                                              edgecolors='r',
                                             label='Acceleration')
                        if info[k] == 2:
                            std = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                             marker='^',c='g',
                                             label='Steady-state')
                        else:
                            continue
                elif plot == 'Heatmap':
                    points   = np.array([t, pos]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm     = plt.Normalize(v_out.min(), v_out.max())
                    lc       = LineCollection(segments, cmap='inferno', norm=norm)
                    lc.set_array(v_out)
                    lc.set_linewidth(2)
                    line     = axs.add_collection(lc)
                else:
                    None
            if plot == 'Normal':        
                for i in range(windowed_lane_lc_ids.size):
                    car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_lc_ids[i])]
                    if car[:,4].size <150:
                        continue
                    car = car[np.where(car[:,5]==j+1)]
                    pos = car[66:-66,2]
                    t   = car[66:-66,1]
                    plt.plot(t,pos,'k', linewidth=1,ls='--')
            else:
                None
        if plot == 'Normal':
            plt.legend((dec, acc, std),
                       ('Deceleration', 'Acceleration'),
                       scatterpoints=1,
                       loc='lower right',
                       ncol=1,
                       fontsize=15)
            plt.xlabel('Time (h)')
            plt.ylabel('Position (km)')
            plt.ylim(ylim_min,ylim_max)
            plt.xlim(window_start,t[-1])
#            plt.savefig('Figures/Dataset_'+str(switch+1)+'/Frame_'+str(l+1)+'_lane_'+lanes[j]+'.png',bbox_inches='tight')
            plt.show()
        if plot == 'Heatmap':
            axs.set_xlabel('Time (h)')
            axs.set_ylabel('Position (km)')
            axs.set_ylim(ylim_min,ylim_max)
            axs.set_xlim(window_start,t[-1])
    #        plt.savefig('Figures/Dataset_'+str(switch+1)+'/Frame_'+str(l+1)+'_lane_'+lanes[j]+'.png',bbox_inches='tight')
            fig.colorbar(lc, ax=axs)
            plt.show()
        else:
            None
        peaks_groups = peak_ids,peaks_time,peaks_pos,peaks_info
        peaks_groups = np.asarray(peaks_groups).T
        
        return peaks_groups
#%%
        
def cluster_peaks(peaks_groups):
    '''
    Cluster oscillations
    
    INPUT: peaks_groups: All peaks detected in the form [Car ID, Peak Time, Peak Position , Peak Type]
    OUTPUT: wave_cluster: List of clusters with peaks inside
    '''
    
    car_ids   = np.unique(peaks_groups[:,0])
    num_peaks = len(peaks_groups[:,0])
    
    first_car_num = np.where(peaks_groups[:,0]==car_ids[0])[0].size  
    
    cntr_cluster  = first_car_num
    wave_clusters = [[] for _ in range(cntr_cluster)]
    
    for i in range(first_car_num):
        wave_clusters[i]=[[peaks_groups[i,0],
                                 peaks_groups[i,1],
                                 peaks_groups[i,2],
                                 peaks_groups[i,3], 
                                 i+1                 ]]
        
    for i in range(first_car_num-1,num_peaks):
        t0  =  peaks_groups[i,1]
        x0  = peaks_groups[i,2]
        id0 = peaks_groups[i,0]
        
        cluster_array = np.zeros((cntr_cluster,3))
        for j in range(cntr_cluster):
            t   = wave_clusters[j][-1][1]
            x   = wave_clusters[j][-1][2]
            cid = wave_clusters[j][-1][0]
            
            [cluster_flag, xt_distance] = clust_assign(t0, x0,id0, t, x, cid)
            
            cluster_array[j,0] = cluster_flag
            cluster_array[j,1] = xt_distance
            cluster_array[j,2] = wave_clusters[j][-1][4]
        
        num_clust_assing = len(np.where(cluster_array[:,0]==1)[0])
        
        if num_clust_assing == 0:
            cntr_cluster = cntr_cluster + 1
            clust_label = cntr_cluster
            temp = np.append(peaks_groups[i,:],clust_label)
            wave_clusters.append([temp.tolist()])
        
        elif num_clust_assing == 1:
            clust_label = int(cluster_array[np.where(cluster_array[:,0]==1)[0],2])
            temp = np.append(peaks_groups[i,:],clust_label)
            wave_clusters[clust_label-1].append(temp.tolist())
            
        elif num_clust_assing > 1:
            indx = np.argmin(cluster_array[:,1])
            clust_label = int(cluster_array[indx,2])
            temp = np.append(peaks_groups[i,:],clust_label)
            wave_clusters[clust_label-1].append(temp.tolist())
            
            
    for i in range(cntr_cluster):
        if len(wave_clusters[i]) <5:
            wave_clusters[i] = [[]]
            
    wave_clusters = [x for x in wave_clusters if x != [[]]]
    
    return wave_clusters


#%%
    
def breakpoint_detect(wave_clusters):
    #breakpoints detection
    n_clust = len(wave_clusters)  
    num_break = 0
    breaks_locs = [[] for _ in range(n_clust)]
    for j in range(n_clust):
        temp = np.asarray(wave_clusters[j])
        slope = np.zeros(len(temp))
        intercept = np.zeros(len(temp))
        r_val = np.zeros(len(temp))
        for i in range(3,len(temp)):
            slope[i], intercept[i], r_value, _t, _k = stats.linregress(temp[:i,1],temp[:i,2])
            r_val[i] = r_value**2/ len(temp[:i,1])
            
            
        loc_break, _d = signal.find_peaks(r_val)
        breaks_locs[j].append(loc_break)
        num_break = num_break +len(loc_break)
        if loc_break.size !=0:
            for i in range(loc_break.size):
                if i==0:
                    plt.scatter(temp[loc_break[i],1],temp[loc_break[i],2],c='green',s=80,label='Breakpoint')    
                    plt.plot(np.unique(temp[:loc_break[i],1]), 
                             np.poly1d(np.polyfit(temp[:loc_break[i],1], 
                                                  temp[:loc_break[i],2], 1))(np.unique(temp[:loc_break[i],1])),linewidth =4)
                else:
                    plt.scatter(temp[loc_break[i],1],temp[loc_break[i],2],c='green',s=80,label='Breakpoint')    
                    plt.plot(np.unique(temp[loc_break[i-1]:loc_break[i],1]), 
                             np.poly1d(np.polyfit(temp[loc_break[i-1]:loc_break[i],1], 
                                                  temp[loc_break[i-1]:loc_break[i],2], 1))(np.unique(temp[loc_break[i-1]:loc_break[i],1])),linewidth =4)
        
            
    #            plt.scatter(temp[loc_break[-1],1],temp[loc_break[-1],2],c='green',s=80)    
                plt.plot(np.unique(temp[loc_break[-1]:,1]), 
                             np.poly1d(np.polyfit(temp[loc_break[-1]:,1], 
                                                  temp[loc_break[-1]:,2], 1))(np.unique(temp[loc_break[-1]:,1])),linewidth =4)
    
        
    return breaks_locs, num_break

######
    