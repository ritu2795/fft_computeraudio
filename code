import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
SAMPLE_RATE = 44100
N=8192
 
def fft(x):
    X = list()
    count =1
    for k in range(0, N):
        window = 1 # np.sin(np.pi * (k+0.5)/N)**2
        X.append(np.complex(x[k] * window, 0))
        count+=1
    fft_rec(X)
    return X
 
def fft_rec(X):
    N = len(X)
    if N <= 1:
        return
 
    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])
 
    fft_rec(even)
    fft_rec(odd)
 
    for k in range(0, N//2):
        t = np.exp(np.complex(0, -2 * np.pi * k / N)) * odd[k]
        X[k] = even[k] + t
        X[N//2 + k] = even[k] - t
 
 
 
df = pd.read_csv('test4.dat', sep='\s+', header=None)
x1=df[1]
xlist1=[]
for i in x1:
    xlist1.append(i)
xlist1=xlist1[:1024]


X = fft(x1)
 
 
# Plotting 
_, plots = plt.subplots(2)
 
## Plot in time domain
#plots[0].plot(x)
## Plot in frequent domain
powers_all = np.abs(np.divide(X, N//2))
powers = powers_all[0:N//2]
frequencies = np.divide(np.multiply(SAMPLE_RATE, np.arange(0, N/2)), N)
frequencies=frequencies[0:512]
powers=powers[0:512]
plots[1].plot(frequencies,powers)
## Show plots
plt.show()

