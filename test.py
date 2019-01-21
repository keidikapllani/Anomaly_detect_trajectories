import numpy as np
import sys
import time
import os
import re 
f = open('data/0750am-0805am/trajectories-0750am-0805am.txt', 'rt')

lines = f.read().split('    ')
import pandas as pd
data = pd.read_csv('data/0750am-0805am/trajectories-0750am-0805am.txt', delim_whitespace=True, header=None)
