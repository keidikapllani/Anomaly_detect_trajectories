# -*- coding: utf-8 -*-
"""
%%%%%%%%%%
@author: kk2314

File to create example signals and perform wavelet energy peak detection as well as printing the results
%%%%%%%%%%
"""

import numpy as np
from utils import  wvlt_ener, peak_classify, load_data
import matplotlib.pyplot as plt
import scipy.io as sio
import pywt
from scipy import signal



#%%
'''
Plotting example singlas to demonstarte the peak detection by cwt
'''


data_smooth, _t = load_data(1)


high_v = 70
low_v = 10
mid_v = 35

half_cycle_length = 64

H = high_v * np.ones(240)
L = low_v * np.ones(240)

D = np.linspace(high_v, low_v, half_cycle_length)
A = np.linspace(low_v, high_v, half_cycle_length)
select = 1
if select==1:
    test_signal = np.concatenate((H,D,A,H), axis=0)
if select==2:
    test_signal = np.concatenate((H, D, A, D, A, H), axis=0)
if select==3:
    test_signal = np.concatenate((L,A,D,H), axis=0)
if select==4:
    test_signal = np.concatenate((L, A, D, A, H), axis=0)

t = np.arange(0,len(test_signal))
#%%

#% ---------------------- The fixed a, b -----------------------------%
a = 32
coef, _t=pywt.cwt(test_signal,a,'mexh')
E = (1/32) * np.sum(np.square(np.abs(coef)), axis=0)

thresh = 0.5e4
locs, _d = signal.find_peaks(E,height=thresh)

t_simple = t[locs]

#% ---------------------- The original method -----------------------------%

e_p_original = wvlt_ener(test_signal)
locs1, _d = signal.find_peaks(e_p_original, height=thresh)

t_original= t[locs1]
#% ------------------------ E1 & E2 method --------------------------------%
test_sign_n = np.max(test_signal) - test_signal

e_p = wvlt_ener(test_signal)
e_n = wvlt_ener(test_sign_n)

locs_p, _d = signal.find_peaks(e_p,height=thresh)
locs_n, _d = signal.find_peaks(e_n,height=thresh)

t_p = t[locs_p]
t_n = t[locs_n]

#% ---------- The complete process (with edge effect reduction------------ %

zr = np.ones(240,)*test_signal[test_signal.size-1]
test_signal_ext = np.append(test_signal,zr)
zr = np.ones(240,)*test_signal[0]
test_signal_ext = np.insert( test_signal_ext, 0, zr)
test_sign_n = np.max(test_signal_ext) - test_signal_ext


e_p_full = wvlt_ener(test_signal_ext)
e_n_full = wvlt_ener(test_sign_n)
#remove extended data
e_p_full = e_p_full[304:-304]
e_n_full = e_n_full[304:-304]
#find peaks and classify data
locs_p, _d = signal.find_peaks(e_p_full,height=thresh)
locs_n, _d = signal.find_peaks(e_n_full,height=thresh)
locs, info = peak_classify(locs_p, locs_n,test_signal[64:-64])

t_er = t[64:-64]
t_final = t_er[locs_p]
t_final_n = t_er[locs_n]


#% ---------- The complete process for a real car velocity signal------------ %
car = data_smooth[np.where(data_smooth[:,0]==756)]
vel = car[1:-1,4]
zr = np.ones(240,)*vel[vel.size-1]
test_signal_ext = np.append(vel,zr)
zr = np.ones(240,)*vel[0]
test_signal_ext = np.insert( test_signal_ext, 0, zr)
test_sign_n = np.max(test_signal_ext) - test_signal_ext


e_p_car = wvlt_ener(test_signal_ext)
e_n_car = wvlt_ener(test_sign_n)
#remove extended data
e_p_car = e_p_car[304:-304]
e_n_car = e_n_car[304:-304]
#find peaks and classify data
locs_car_p, _d = signal.find_peaks(e_p_car,height=thresh)
locs_car_n, _d = signal.find_peaks(e_n_car,height=thresh)
locs, info = peak_classify(locs_car_p, locs_car_n,vel[64:-64])

t_er = car[64:-64,1]
t_final = t_er[locs_car_p]
t_final_n = t_er[locs_car_n]


#%%%
'''
Plotting mexican wavelet
'''


 
points = np.arange(-50,51)
a = 4.0
vec2 = signal.ricker(101, a)
plt.plot(points,vec2,linewidth=2)
plt.xlim(-30,30)
plt.ylabel('ψ(t)')
plt.xlabel('Time')

plt.show()
 
 
#%%%
'''
Plotting prototype signal together with eb1 and eb2 energy signals together with anotated peaks

'''
fig, ax = plt.subplots(3, 1,gridspec_kw={'hspace': 0.3, 'wspace': 0.3},figsize=(5,2.5))

ax[0].plot(t,test_signal,'b', linewidth=1.5,label='Prototype Signal')
ax[0].set(ylabel='Velocity (km/h)', )
ax[0].set_xlim(0, len(test_signal))
ax[0].legend()

ax[1].plot(t,e_p,'b', linewidth=1.5,label='Eb1')
n = np.arange(1,len(t_p)+1)
ax[1].scatter(t_p,e_p[t_p], s=80, facecolors='none', edgecolors='r')
ax[1].set(ylabel='ψ(t)', )
ax[1].set_xlim(0,  len(test_signal))
ax[1].legend()
for i, txt in enumerate(n):
    ax[1].annotate(txt, (t_p[i],e_p[t_p[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')


ax[2].plot(t,e_n,'b', linewidth=1.5,label='Eb2')
n = np.arange(1,len(t_n)+1)
ax[2].scatter(t_n,e_n[t_n], s=80, facecolors='none', edgecolors='r')
ax[2].set(ylabel='ψ(t)', xlabel='Time')
ax[2].set_xlim(0,  len(test_signal))
ax[2].legend()
for i, txt in enumerate(n):
    ax[2].annotate(txt, (t_n[i],e_n[t_n[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')

fig.set_dpi(150)
plt.subplots_adjust(top=0.99,
                    bottom=0.085,
                    left=0.085,
                    right=0.99,
                    hspace=0.2,
                    wspace=0.2)
#%%
'''
Plotting prototype signal together with original and edge reduced eb1
'''

fig, ax = plt.subplots(3, 1,gridspec_kw={'hspace': 0.3, 'wspace': 0.3},figsize=(5,2.5))

ax[0].plot(t,test_signal,'b', linewidth=1.5,label='Prototype Signal')
ax[0].set(ylabel='Velocity (km/h)', )
ax[0].set_xlim(0, len(test_signal))
ax[0].legend()

ax[1].plot(t_er,e_p_full,'b', linewidth=1.5,label='Eb1')
n = np.arange(1,len(t_final)+1)
ax[1].scatter(t_final,e_p_full[locs_p], s=80, facecolors='none', edgecolors='r')
ax[1].set(ylabel='ψ(t)', )
ax[1].set_xlim(0,  len(test_signal))
ax[1].legend()
for i, txt in enumerate(n):
    ax[1].annotate(txt, (t_final[i],e_p_full[locs_p[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')


ax[2].plot(t_er,e_n_full,'b', linewidth=1.5,label='Eb2')
n = np.arange(1,len(t_final_n)+1)
ax[2].scatter(t_final_n,e_n_full[locs_n], s=80, facecolors='none', edgecolors='r')
ax[2].set(ylabel='ψ(t)', xlabel='Time')
ax[2].set_xlim(0,  len(test_signal))
ax[2].legend()
for i, txt in enumerate(n):
    ax[2].annotate(txt, (t_final_n[i],e_n_full[locs_n[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')

fig.set_dpi(150)
plt.subplots_adjust(top=0.99,
                    bottom=0.085,
                    left=0.085,
                    right=0.99,
                    hspace=0.2,
                    wspace=0.2)

#%%
'''
Plotting real data and eb1 and eb2
'''

fig, ax = plt.subplots(3, 1,gridspec_kw={'hspace': 0.3, 'wspace': 0.3},figsize=(5,2.5))

ax[0].plot(vel[64:-64],'b', linewidth=1.5,label='Speed data')
ax[0].set(ylabel='Velocity (km/h)', )
ax[0].set_xlim(0, len(vel[64:-64]))
ax[0].legend()

ax[1].plot(e_p_car,'b', linewidth=1.5,label='Eb1')
n = np.arange(1,len(t_final)+1)
ax[1].scatter(locs_car_p,e_p_car[locs_car_p], s=80, facecolors='none', edgecolors='r')
ax[1].set(ylabel='ψ(t)', )
ax[1].set_xlim(0,  len(vel[64:-64]))
ax[1].legend()
for i, txt in enumerate(n):
    ax[1].annotate(txt, (locs_car_p[i],e_p_car[locs_car_p[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')
    
ax[2].plot(e_n_car,'b', linewidth=1.5,label='Eb2')
n = np.arange(1,len(t_final_n)+1)
ax[2].scatter(locs_car_n,e_n_car[locs_car_n], s=80, facecolors='none', edgecolors='r')
ax[2].set(ylabel='ψ(t)', xlabel='Time')
ax[2].set_xlim(0,  len(e_n_car))
ax[2].legend()
for i, txt in enumerate(n):
    ax[2].annotate(txt, (locs_car_n[i],e_n_car[locs_car_n[i]]),textcoords="offset points", # how to position the text
                 xytext=(-10,2), # distance from text to points (x,y)
                 ha='center')


fig.set_dpi(150)
plt.subplots_adjust(top=0.99,
                    bottom=0.085,
                    left=0.085,
                    right=0.99,
                    hspace=0.2,
                    wspace=0.2)

#%%
plt.figure()
plt.plot(vel,'k', linewidth=1.5,label='Speed data')
plt.ylabel('Velocity (km/h)' )
plt.xlabel('Time (h)')
plt.xlim(0, len(vel))
plt.scatter(locs_car_p+64,vel[locs_car_p+64], s=80, facecolors='none', edgecolors='r', label='Deceleration')
plt.scatter(locs_car_n+64,vel[locs_car_n+64], s=80, facecolors='none', edgecolors='b', label='Acceleration')
plt.legend()





























