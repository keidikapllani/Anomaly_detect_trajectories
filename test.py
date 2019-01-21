import numpy as np
import pandas as pd
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

data_first = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
data_first.columns = ['Vehicle ID','Frame ID','Total Frames','Global Time','Local X','Local Y','Global X', 'Global Y','Vehicle Length','Vehicle Width', 'Vehicle Class', 'Vehicle Velocity', 'Vehicle Acceleration', 'Lane Identification', 'Preceding Vehicle', 'Following Vehicle', 'Spacing', 'Headway ']
data_first['Vehicle ID'].unique().size