from __future__ import division, print_function, absolute_import
import numpy as np
<<<<<<< HEAD
from utils import load_data, lane_changer, wvlt_ener, peak_classify, clust_assign
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import signal
from sklearn.cluster import SpectralClustering
from itertools import cycle
from scipy import stats 
=======
from utils import load_data, lane_changer, wvlt_ener, peak_classify
import matplotlib.pyplot as plt

from scipy import signal

>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
#%matplotlib auto


#%%

<<<<<<< HEAD
switch = 0
data_smooth, _p = load_data(switch,False)
=======
switch = 1
data_smooth, _p = load_data(switch,True)
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786

#%%
'''
Finding ids of cars in lane j frame l

'''

<<<<<<< HEAD
for l in range(1):
    #find data in time windows
    car_ids = np.unique(data_smooth[:,0]).astype(int)
    frame_length = 15/60
    window_start = data_smooth[0,1] + frame_length*l 
    window_end = window_start + frame_length
#    window_start= 7.83
#    window_end = 7.88
=======
for l in range(0,1):
    #find data in time windows
    car_ids = np.unique(data_smooth[:,0]).astype(int)
    frame_length = 5/60
    window_start = data_smooth[0,1] + frame_length*l
    window_end = window_start + frame_length
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
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
    
<<<<<<< HEAD
    for j in range(4,5):    
=======
    for j in range(1):
    
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
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
<<<<<<< HEAD
        fig, axs = plt.subplots(1,1,figsize=(15,10))
#        plt.figure(figsize=(15,10))
=======
        plt.figure(figsize=(15,10))
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
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
<<<<<<< HEAD
            thresh = 1e4
=======
            thresh = 0.4e4
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
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
                
           
            
<<<<<<< HEAD
#            plt.plot(t,pos,'k', linewidth=1) # normal plot
#            
#            plot with heat map
#            points = np.array([t, pos]).T.reshape(-1, 1, 2)
#            segments = np.concatenate([points[:-1], points[1:]], axis=1)
#            
#          
#            
#            norm = plt.Normalize(v_out.min(), v_out.max())
#            lc = LineCollection(segments, cmap='inferno', norm=norm)
#            # Set the values used for colormapping
#            lc.set_array(v_out)
#            lc.set_linewidth(2)
#            line = axs.add_collection(lc)
#            
#            for k in range(0,info.size):
#                if info[k] == 1:
#                    dec = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                                      label='Deceleration',s=60, 
#                                      facecolors='none', 
#                                      edgecolors='b')
#                if info[k] == 0:
#                    acc = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                                       s=60, 
#                                      facecolors='none', 
#                                      edgecolors='r',
#                                     label='Acceleration')
#                if info[k] == 2:
#                    std = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                                     marker='^',c='g',
#                                     label='Steady-state')
#                else:
#                    continue
#        
#        for i in range(windowed_lane_lc_ids.size):
#            car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_lc_ids[i])]
#            if car[:,4].size <150:
#                continue
#            car = car[np.where(car[:,5]==j+1)]
#            pos = car[66:-66,2]
#            t = car[66:-66,1]
#            
#            plt.plot(t,pos,'k', linewidth=1,ls='--')
        
#        plt.legend((dec, acc, std),
#                   ('Deceleration', 'Acceleration'),
#                   scatterpoints=1,
#                   loc='lower right',
#                   ncol=1,
#                   fontsize=15)
#        axs.set_xlabel('Time (h)')
#        axs.set_ylabel('Position (km)')
##        axs.title('', fontsize= 20)
#        axs.set_ylim(ylim_min,ylim_max)
#        axs.set_xlim(window_start,window_end)
##        plt.savefig('Figures/Dataset_'+str(switch+1)+'/Frame_'+str(l+1)+'_lane_'+lanes[j]+'.png',bbox_inches='tight')
#        plt.show()
#        fig.colorbar(lc, ax=axs)
=======
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
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
        #
        
        peaks_groups = peak_ids,peaks_time,peaks_pos,peaks_info
        peaks_groups = np.asarray(peaks_groups).T
    

<<<<<<< HEAD
=======
#%%
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
car_ids = np.unique(peak_ids)
num_peaks = len(peaks_info)

first_car_num = np.where(peak_ids==car_ids[0])[0].size  

cntr_cluster  = first_car_num
<<<<<<< HEAD
wave_clusters = [[] for _ in range(cntr_cluster)]

for i in range(first_car_num):
    wave_clusters[i]=[[peaks_groups[i,0],
                             peaks_groups[i,1],
                             peaks_groups[i,2],
                             peaks_groups[i,3], 
                             i+1                 ]]
    


for i in range(first_car_num-1,num_peaks):
    t0 = peaks_time[i]
    x0 = peaks_pos[i]
    id0 = peak_ids[i]
    
    cluster_array = np.zeros((cntr_cluster,3))
    for j in range(cntr_cluster):
        t = wave_clusters[j][-1][1]
        x = wave_clusters[j][-1][2]
        cid = wave_clusters[j][-1][0]
        
        [cluster_flag, xt_distance] = clust_assign(t0, x0,id0, t, x, cid);
        
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

n_clust = len(wave_clusters)  

#breakpoints detection
num_break = 0
for j in range(len(wave_clusters)):
    temp = np.asarray(wave_clusters[j])
    slope = np.zeros(len(temp))
    intercept = np.zeros(len(temp))
    r_val = np.zeros(len(temp))
    for i in range(3,len(temp)):
        slope[i], intercept[i], r_value, _t, _k = stats.linregress(temp[:i,1],temp[:i,2])
        r_val[i] = r_value**2/ len(temp[:i,1])
        
        
    loc_break, _d = signal.find_peaks(r_val)
    num_break = num_break +len(loc_break)

######

num_a_peaks = len(peaks_info[np.where(peaks_info==1)])
num_d_peaks = len(peaks_info[np.where(peaks_info==0)])
num_clusters = len(wave_clusters)
print(' A peaks = '+str(num_a_peaks)+' D peaks = '+str(num_d_peaks)+' number of clusters= '+str(num_clusters)+ ' number of breaks= '+str(num_break ))





        


        #%%
#Plot clusters
n_clust = len(wave_clusters)     
cmap = plt.cm.get_cmap("tab10", n_clust) 
cycol = cycle('bgrcmk')
for i in range(n_clust):
    temp = np.asarray(wave_clusters[i])
#    one_clust = np.vstack([one_clust,temp])
    plt.scatter(temp[:,1],temp[:,2],c=next(cycol), label= "Cluster "+str(i+1))
    
    
            
        
        
#plt.scatter(one_clust[:,1],one_clust[:,2],c=one_clust[:,4])
  
plt.legend()
plt.show()        
#%%
'''
K-means clustering for osciallations tracing
'''
=======
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

>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
from sklearn.cluster import KMeans
from sklearn import preprocessing as prep


#plt.scatter(peaks_time,peaks_pos)
index_peaks = np.where(peaks_info != 2)[0]
alfa_test = peaks_time,peaks_pos,peaks_info
<<<<<<< HEAD
alfa_test = np.zeros((328,3))
=======
alfa_test = np.zeros((114,3))
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
alfa_test[:,0] = prep.normalize([peaks_time[index_peaks]])
alfa_test[:,1] = prep.normalize([peaks_pos[index_peaks]])
alfa_test[:,2] = prep.normalize([peaks_info[index_peaks]])

<<<<<<< HEAD

clustering = SpectralClustering(n_clusters=11, assign_labels="discretize", random_state=0).fit(alfa_test)


alfa_vals= clustering.labels_
#plt.figure()
plt.scatter(peaks_time[index_peaks], peaks_pos[index_peaks], c=clustering.labels_)
plt.legend()

#%%%
''''
Breakpoint Detection
'''

#from sklearn import linear_model

#temp = np.asarray(wave_clusters[0])

for j in range(len(wave_clusters)):
    temp = np.asarray(wave_clusters[j])
    slope = np.zeros(len(temp))
    intercept = np.zeros(len(temp))
    r_val = np.zeros(len(temp))
    for i in range(3,len(temp)):
        slope[i], intercept[i], r_value, _t, _k = stats.linregress(temp[:i,1],temp[:i,2])
        r_val[i] = r_value**2/ len(temp[:i,1])
        
        
    loc_break, _d = signal.find_peaks(r_val)
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
    
    

plt.show()    
    
    #%%
plt.figure()
for i in range(0,windowed_lane_cf_ids.size):
    
    #get a single vehilce
    car = data_smooth[np.where(data_smooth[:,0]==windowed_lane_cf_ids[i])]
    
    if car[:,4].size <150:
        continue
#    print(car[:,4].size)
    vel = car[66:-66,4]
    v_out = vel[64:-64]
    pos = car[-66:66,2]
    t = car[-66:66,1]
    plt.plot(t,pos,'k', linewidth=1) # normal plot
#  
#    for k in range(0,info.size):
#        if info[k] == 1:
#            dec = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                              label='Deceleration',s=60, 
#                              facecolors='none', 
#                              edgecolors='b')
#        if info[k] == 0:
#            acc = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                               s=60, 
#                              facecolors='none', 
#                              edgecolors='r',
#                             label='Acceleration')
#        if info[k] == 2:
#            std = plt.scatter(t[locs[k]+2],pos[locs[k]+2],
#                             marker='^',c='g',
#                             label='Steady-state')
#        else:
#            continue


#plt.legend((dec, acc, std),
#           ('Deceleration', 'Acceleration'),
#           scatterpoints=1,
#           loc='lower right',
#           ncol=1,
#           fontsize=15)
plt.xlabel('Time (h)')
plt.ylabel('Position (km)')
plt.title('', fontsize= 20)
plt.ylim(ylim_min,ylim_max)
#plt.savefig('Figures/Dataset_'+str(switch+1)+'/Frame_'+str(l+1)+'_lane_'+lanes[j]+'.png',bbox_inches='tight')
plt.show()
#
    
    
#%%
plt.scatter(temp[:,1],temp[:,2])
plt.scatter(temp[12,1],temp[12,2],c='green',s=80)    
    
r_value = - r_value
    
break_loc, _d = signal.find_peaks(r_value,height=0.52)
    
    
    
#from seg_reg import SegmentedLinearReg    
#
#plt.scatter( temp[:,1], temp[:,2] )
#
#initialBreakpoints = [1]
#plt.plot( *SegmentedLinearReg( temp[:,1], temp[:,2], initialBreakpoints ), '-r' );
#plt.xlabel('X'); plt.ylabel('Y');
#
#def piecewise_linear(x, x0, y0, k1, k2):
#    return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
#    
#from scipy import optimize
#p , e = optimize.curve_fit(piecewise_linear, temp[:,1], temp[:,2])
#plt.plot(x, y, "o")
#plt.plot(xd, piecewise_linear(xd, *p))
#
#
#import pwlf
#
#    
#    
#    
#    
#    
#    
#from mlinsights.mlmodel import PiecewiseRegressor
#from sklearn.tree import DecisionTreeRegressor
#
#model = PiecewiseRegressor(verbose=True,
#                           binner=DecisionTreeRegressor(min_samples_leaf=300))
#model.fit(temp[:,1], temp[:,2])
#    
#    
    
    #alfa = np.zeros(len(temp))
#for i in range(1,len(temp)):
#    reg.fit(temp[:i,1].reshape(-1,1),temp[:i,2])
#    alfa[i] =  reg.coef_

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
=======
kmeans = KMeans(n_clusters=4, random_state=0).fit(alfa_test)
alfa_vals= kmeans.labels_

plt.scatter(peaks_time[index_peaks], peaks_pos[index_peaks], c=kmeans.labels_)
>>>>>>> 2a4a234659285a33d329c03ade40355c002dd786
