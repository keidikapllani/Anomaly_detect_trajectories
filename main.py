import numpy as np
from utils import load_data, anomaly_detect, cluster_peaks, breakpoint_detect
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import SpectralClustering
from itertools import cycle
from scipy import stats 
#matplotlib auto


#%%
'''
Load Data

    switch: 0 - 0750am-0805am
            1 - 0805am-0820am
            2 - 0820am-0835am

'''
switch = 0
data_smooth, _p = load_data(switch,False)



#%%
'''
Finding ids of cars in lane j frame l

'''

peaks_groups = anomaly_detect(data_smooth, frame_length =3/60, plot = 'Normal', start_frame = 0, num_frames = 1, start_lane = 0, num_lanes=1 )
    

#%%


wave_clusters = cluster_peaks(peaks_groups)


breaks_locs, num_break = breakpoint_detect(wave_clusters)


num_a_peaks = len(peaks_groups[np.where(peaks_groups[:,3]==1)])
num_d_peaks = len(peaks_groups[np.where(peaks_groups[:,3]==0)])
num_clusters = len(wave_clusters)
print(' A peaks = '+str(num_a_peaks)+' D peaks = '+str(num_d_peaks)+' number of clusters= '+str(num_clusters)+ ' number of breaks= '+str(num_break ))



#%%
'''
K-means clustering for osciallations tracing
'''
from sklearn.cluster import KMeans
from sklearn import preprocessing as prep


#plt.scatter(peaks_time,peaks_pos)
index_peaks = np.where(peaks_info != 2)[0]
alfa_test = peaks_time,peaks_pos,peaks_info
alfa_test = np.zeros((328,3))
alfa_test[:,0] = prep.normalize([peaks_time[index_peaks]])
alfa_test[:,1] = prep.normalize([peaks_pos[index_peaks]])
alfa_test[:,2] = prep.normalize([peaks_info[index_peaks]])


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
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
