# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv

with open('Data/Results16.csv') as file2:
    csv_reader = csv.reader(file2, delimiter=',')
    line_count = 0
    data = []
    for i in csv_reader:
        if (line_count == 0):
            parameters = list(map(lambda x: float(x),i))
            line_count += 1
        else:
            data.append(np.array(list(map(lambda x: float(x),i[:-1]))))
            line_count += 1
            
tiempos = [parameters[1]*i for i in range(0,int(parameters[0]))]
xrange = np.linspace(-3,3,100)

def histogram(data,xrange):
    deltax = 0.5*(xrange[1]-xrange[0])  # Anchura de cada bin
    N = len(data)*len(data[0])
    xcenters = [i+deltax for i in xrange[:-1]] # Centro de cada bin
    binvalues = []
    for i in range(0,len(xcenters)):
        count = 0
        for j in data:
            for k in j:
                if ((k > xcenters[i] - deltax) and (k <= xcenters[i] + deltax)):
                    count += 1
        binvalues.append(count/(N*2*deltax))
    return ([xcenters,binvalues])

def probexp(x,data,xrange):
    deltax = 0.1
    N = len(data)
    count = 0
    for i in data:
        if (abs(x-i) < deltax): 
            count += 1
    return count/N
    
def probfunction(x):
    return 0.59*np.exp(-1.1*x**2)

plt.plot(data[0],tiempos)
plt.show()

xvalues, yvaluesexp = histogram(data,xrange)
yvaluestheo = [probfunction(i) for i in xrange]
plt.plot(xrange, yvaluestheo, '--', color = 'blue')
plt.plot(xvalues, yvaluesexp, '-', color = 'red')
plt.show()
