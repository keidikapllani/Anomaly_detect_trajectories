
import numpy as np
import pandas as pd
import scipy.io as sio
import pywt
from utils import *
import matplotlib.pyplot as plt

#%%
#pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

data_first  = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
data_second = pd.read_csv('data/0805am-0820am/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
data_third  = pd.read_csv('data/0820am-0835am/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
data_first.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
data_second.columns = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
data_third.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
#
#vehicle_number_first = data_first['Vehicle ID'].unique().size
#vehicle_number_second = data_second['Vehicle ID'].unique().size
#vehicle_number_third = data_third['Vehicle ID'].unique().size

#keidi = data_first.groupby('Vehicle ID')

data_raw = data_first.values  #+ data_second.values + data_third.values
keep = [0,3,6,10,11,13,16]
data_raw = data_raw[:,keep]
data_raw[:,1] = (data_raw[:,1]-data_raw[0,1])/1000 #fix time to ms since start of observation
data_raw[:,2] = data_raw[:,2]* 0.0003048 #convert to km
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
c;l = pywt.wavedec(da)