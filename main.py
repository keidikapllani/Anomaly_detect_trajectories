import numpy as np
from utils import load_data, anomaly_detect, cluster_peaks
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

peaks_groups = anomaly_detect(data_smooth, frame_length =1.5/60, plot = 'Normal', start_frame = 0, end_frame = 1, start_lane = 0, end_lane=1 )
    

#%%

wave_clusters = cluster_peaks(peaks_groups)

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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
