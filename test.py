import numpy as np
import pandas as pd
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

data_first  = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
data_second = pd.read_csv('data/0805am-0820am/trajectories-0805am-0820am.txt', delim_whitespace=True, header=None)
data_third  = pd.read_csv('data/0820am-0835am/trajectories-0820am-0835am.txt', delim_whitespace=True, header=None)
data_first.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
data_second.columns = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
data_third.columns  = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']

vehicle_number_first = data_first['Vehicle ID'].unique().size
vehicle_number_second = data_second['Vehicle ID'].unique().size
vehicle_number_third = data_third['Vehicle ID'].unique().size

