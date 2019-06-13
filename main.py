from __future__ import division, print_function, absolute_import
import numpy as np
from utils import load_data, lane_changer, wvlt_ener, peak_classify
import matplotlib.pyplot as plt

from scipy import signal

#%matplotlib auto


#%%

switch = 1
data_smooth, _p = load_data(switch,True)

#%%
'''
Finding ids of cars in lane j frame l

'''

for l in range(0,1):
    #find data in time windows
    car_ids = np.unique(data_smooth[:,0]).astype(int)
    frame_length = 5/60
    window_start = data_smooth[0,1] + frame_length*l
    window_end = window_start + frame_length
    windowed_car_ids = []
    for i in range(len(car_ids)):
        car = data_smooth[np.where(data_smooth[:,0]==car_ids[i])]
        car_start_t = car[0,1]
        if (car_start_t > window_start and car_start_t<window_end):
            windowed_car_ids = np.append(windowed_car_ids,car[0,0])
        else:
            continue
    lane_nor_id, lane_chang_id = lane_changer(data_smooth)
    lanes = ['1','2','3','4','5']
    peaks_groups = []
    
    for j in range(1):
    
        print('LOG: Working lane '+lanes[j]+' in frame '+str(l+1))
        #find ids in lane j+1 
        
        idx_lane1 = np.where(data_smooth[:,5]==j+1)
        lane_1 = data_smooth[idx_lane1[0],:]
        ids_lane_1 = np.unique(lane_1[:,0]).astype(int)
        #find non-lane changers in frame
        no_change_time_wind_ids = np.intersect1d(lane_nor_id,windowed_car_ids).astype(int)
        #find lane changers in framce 
        lane_cngr_time_wind_ids = np.intersect1d(lane_chang_id,windowed_car_ids).astype(int)
        #find non-lane changers in lane i+1 in frame
        windowed_lane_cf_ids = np.intersect1d(no_change_time_wind_ids,ids_lane_1).astype(int)
        #find non-lane changers in lane i+1 in frame 
        windowed_lane_lc_ids = np.intersect1d(lane_cngr_time_wind_ids,ids_lane_1).astype(int)
        car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[0])]
        ylim_min = car[66,2]  
        ylim_max = car[len(car)-67,2]
        plt.figure(figsize=(15,10))
        for i in range(0,windowed_lane_cf_ids.size):
            #get a single vehilce
            car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[i])]
            
            if car[:,4].size <150:
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
            thresh = 0.4e4
            locs_p, _d = signal.find_peaks(e_p,height=thresh)
            locs_n, _d = signal.find_peaks(e_n,height=thresh)
            locs, info = peak_classify(locs_p, locs_n,v_out)
            if i == 0:
                peak_ids = np.repeat(car[0,0],info.size)
                peaks_time=t[locs]
                peaks_pos = pos[locs]
                peaks_info = info
            else:
                peak_ids = np.append(peak_ids,np.repeat(car[0,0],info.size))
                peaks_time = np.append(peaks_time,t[locs])
                peaks_pos= np.append(peaks_pos,pos[locs])
                peaks_info = np.append(peaks_info,info)
                
           
            
    #        peaks_groups.append([])
            plt.plot(t,pos,'k', linewidth=1)
            for k in range(0,info.size):
                if info[k] == 1:
                    dec = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                      marker='x',c='blue',
                                      label='Deceleration')
                if info[k] == 0:
                    acc = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                     marker='o',c='red',
                                     label='Acceleration')
                if info[k] == 2:
                    std = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
                                     marker='^',c='green',
                                     label='Stredy-state')
                else:
                    continue
        
        for i in range(windowed_lane_lc_ids.size):
            car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_lc_ids[i])]
            if car[:,4].size <150:
                continue
            car = car[np.where(car[:,5]==j+1)]
            pos = car[66:-66,2]
            t = car[66:-66,1]
            
            plt.plot(t,pos,'k', linewidth=1,ls='--')
        
        plt.legend((dec, acc, std),
                   ('Deceleration', 'Acceleration', 'Steady-state'),
                   scatterpoints=1,
                   loc='lower right',
                   ncol=1,
                   fontsize=15)
        plt.xlabel('Time (h)')
        plt.ylabel('Position (km)')
        plt.title('Anomaly detection demonstration for '+str(l+1)+' frame in lane '+lanes[j], fontsize= 20)
        plt.ylim(ylim_min,ylim_max)
        plt.savefig('Figures/Dataset_'+str(switch+1)+'/Frame_'+str(l+1)+'_lane_'+lanes[j]+'.png',bbox_inches='tight')
        plt.show()
        #
        
        peaks_groups = peak_ids,peaks_time,peaks_pos,peaks_info
        peaks_groups = np.asarray(peaks_groups).T
    

#%%
car_ids = np.unique(peak_ids)
num_peaks = len(peaks_info)

first_car_num = np.where(peak_ids==car_ids[0])[0].size  

cntr_cluster  = first_car_num
wave_clusters = []
for i in range(first_car_num):
    wave_clusters.append([peaks_groups[i,0],
                         peaks_groups[i,1],
                         peaks_groups[i,2],
                         peaks_groups[i,3], 
                         i                 ])
    


for i in range(first_car_num,num_peaks):
    t0 = peaks_time(i);
    x0 = peaks_pos(i);
    id0 = peak_ids(i);
    
    cluster_array = np.zeros(cntr_cluster,3)
    for j in range(cntr_cluster):
        t = wave_clusters[j][1]
        x = wave_clusters[j][2]
        cid = wave_clusters[j][0]
        
#%%

from sklearn.cluster import KMeans
from sklearn import preprocessing as prep


#plt.scatter(peaks_time,peaks_pos)
index_peaks = np.where(peaks_info != 2)[0]
alfa_test = peaks_time,peaks_pos,peaks_info
alfa_test = np.zeros((114,3))
alfa_test[:,0] = prep.normalize([peaks_time[index_peaks]])
alfa_test[:,1] = prep.normalize([peaks_pos[index_peaks]])
alfa_test[:,2] = prep.normalize([peaks_info[index_peaks]])

kmeans = KMeans(n_clusters=4, random_state=0).fit(alfa_test)
alfa_vals= kmeans.labels_

plt.scatter(peaks_time[index_peaks], peaks_pos[index_peaks], c=kmeans.labels_)
