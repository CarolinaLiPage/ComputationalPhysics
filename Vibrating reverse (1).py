#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.integrate import odeint
from numpy import linspace,array,zeros,log,exp,sin,cos,sqrt,pi,e,arange, real, imag, sign
from matplotlib.pyplot import plot,xlabel,ylabel,legend,show, figure, subplot, title, tight_layout, stem
from scipy.fftpack import fft
from pylab import xlim
import numpy as np
from numpy.linalg import eig
import math

# time dimension

Fs = 500 # Hz  sampling frequency: samples per second
dT =  1/Fs # sec   time betfreqeen samples
nt  = 2000 #  2**11 #     number of samples in record, power of 2
T =  nt*dT # Time period of record

t=arange(0,T,dT)  #  time array in seconds using arange(start,stop,step)
            #   note that arange actually stops *before* stop which
            #   is what we want (in a periodic function t=0 ant t=T are the same)

# frequency dimension

freqf =  1/T # Hz   fundamental frequency (lowest frequency)
nfmax = int(nt/2) # number of frequencies resolved by FFT

freqmax = freqf*nfmax # Max frequency (Nyquist)

freq = arange(0,freqmax,freqf) # frequency array using arange(start,stop,step)
 # Note that since we are including freq=0 (constant term), this actually truncates before one
 # term before the term at the Nyquist (max) frequency. 

print('Fundamental period and Nyquist Freq',T, freqmax)

# Create a time series by summing sine and cosine waves

# select four frequencies that are small (<20) multiples of fundamental frequency above
f1=40*freqf
f2=60*freqf
f3=20*freqf
f4=90*freqf

print('Frequencies selected:', f1, f2, f3, f4)

# Create f(t) = A sin(f1 2 pi t) + B sin(f2 2 pi t)  + C cos(f3 2 pi t) + D cos(f4 2 pi t) + E% 
# Where A B C D E are integers between -10 and + 10
# Use your time array defined above so f is an array
# with values at each of these times.
# what is the equation for a sine or cos wave with frequency w1?
#    this is an array over all values of time array t, not a "function"
#    sin and cos can opperate on an array so no loop needed
#    remember factor of 2pi
#    basically, just write it like math!
f =  3 + 4*cos(f1 * 2*pi*t) +  3*cos(f3 * 2*pi*t) + 8*sin(f2 * 2*pi*t) + 2*sin(f4 * 2*pi*t)


# take FFT
F = fft(f)

# get the coeffs
a = 2*real(F[:nfmax])/nt # form the a coefficients
a[0] = a[0]/2

b = -2*imag(F[:nfmax])/nt # form the b coefficients

p = sqrt(a**2 + b**2) # form power spectrum

## make some plots

figure(1)

subplot(2,1,1)
plot(t,f)
title('Signal')

subplot(2,1,2)
plot(freq, a, 'o', label='Cosine')
plot(freq, b, '*', label='Sine')
plot(freq, p,'-', label='Power')
legend() 

title('FFT Fourier Coefficients')
xmax = max([f1, f2, f3, f4])*1.15 # find max value and pad a bit (15%)
xlim(0, xmax)

tight_layout() # prevent squished plot (matplotlib kludge)


# In[30]:


from scipy.io import loadmat
infile = loadmat("/Desktop/VibratingMass_Reverse/Case_1.mat")
t = infile['t'][0,:]
y = infile['y'][0,:]
plot(t,y)
title('Signal')
print(len(t))
T = t[len(t)-1]
nt = 1000
dT = T/nt
Fs = 1/dT

# t0=arange(0,T,dT)  #  time array in seconds using arange(start,stop,step)
            #   note that arange actually stops *before* stop which
            #   is what we want (in a periodic function t=0 ant t=T are the same)

# frequency dimension

freqf =  1/T # Hz   fundamental frequency (lowest frequency)
nfmax = int(nt/2) # number of frequencies resolved by FFT

freqmax = freqf*nfmax # Max frequency (Nyquist)

freq = arange(0,freqmax,freqf) # frequency array using arange(start,stop,step)
 # Note that since we are including freq=0 (constant term), this actually truncates before one
 # term before the term at the Nyquist (max) frequency. 

print('Fundamental period and Nyquist Freq',T, freqmax)

# take FFT
F = fft(y)

# get the coeffs
a = 2*real(F[:nfmax])/nt # form the a coefficients
a[0] = a[0]/2

b = -2*imag(F[:nfmax])/nt # form the b coefficients

p = sqrt(a**2 + b**2) # form power spectrum

## make some plots

figure(1)

subplot(2,1,1)
plot(t,y)
title('Signal')

subplot(2,1,2)
plot(freq, a, 'o', label='Cosine')
plot(freq, b, '*', label='Sine')
plot(freq, p,'-', label='Power')
legend() 

tight_layout() # prevent squished plot (matplotlib kludge)

import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
peaks, _ = find_peaks(p, height=0)
print(peaks/100)
figure(2)
plt.plot(p)
plt.plot(peaks, p[peaks], "x")
plt.plot(np.zeros_like(p), "--", color="gray")
plt.show()

eigen_values = [-x*x for x in peaks/100]
print(eigen_values)


# In[31]:


from scipy.io import loadmat
infile = loadmat("/Desktop/VibratingMass_Reverse/Case_2.mat")
t = infile['t'][0,:]
y = infile['y'][0,:]
plot(t,y)
title('Signal')
print(len(t))
T = t[len(t)-1]
nt = 1000
dT = T/nt
Fs = 1/dT
# frequency dimension

freqf =  1/T # Hz   fundamental frequency (lowest frequency)
nfmax = int(nt/2) # number of frequencies resolved by FFT

freqmax = freqf*nfmax # Max frequency (Nyquist)

freq = arange(0,freqmax,freqf) # frequency array using arange(start,stop,step)
 # Note that since we are including freq=0 (constant term), this actually truncates before one
 # term before the term at the Nyquist (max) frequency. 

print('Fundamental period and Nyquist Freq',T, freqmax)

# take FFT
F = fft(y)

# get the coeffs
a = 2*real(F[:nfmax])/nt # form the a coefficients
a[0] = a[0]/2

b = -2*imag(F[:nfmax])/nt # form the b coefficients

p = sqrt(a**2 + b**2) # form power spectrum

## make some plots

figure(1)

subplot(2,1,1)
plot(t,y)
title('Signal')

subplot(2,1,2)
plot(freq, a, 'o', label='Cosine')
plot(freq, b, '*', label='Sine')
plot(freq, p,'-', label='Power')
legend() 

tight_layout() # prevent squished plot (matplotlib kludge)

import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
peaks, _ = find_peaks(p, height=0)
print(peaks/100)
figure(2)
plt.plot(p)
plt.plot(peaks, p[peaks], "x")
plt.plot(np.zeros_like(p), "--", color="gray")
plt.show()

eigen_values = [-x*x for x in peaks/100]
print(eigen_values)



