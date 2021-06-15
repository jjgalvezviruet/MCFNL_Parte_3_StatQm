# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv

with open('Probdata/Results12.csv') as file2:
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
            
""" Rangos en el eje x"""
xrange = np.linspace(-1,1,30)
print(len(data)*len(data[0]))

deltax = 0.5*(xrange[1]-xrange[0])
xcenters = [i+deltax for i in xrange[:-1]] # Centros de bins en histogramas

def totalhistogram(data,xrange):
    deltax = 0.5*(xrange[1]-xrange[0])  # Anchura de cada bin
    N = len(data)*len(data[0])
    binvalues = []
    binerrors = []
    for i in range(0,len(xcenters)):
        count = 0
        for j in data:
            for k in j:
                if ((k > xcenters[i] - deltax) and (k <= xcenters[i] + deltax)):
                    count += 1
        binvalues.append(count/(N*2*deltax))
        binerrors.append(np.sqrt(count/(2*deltax))/N)
    return ([binvalues,binerrors])


def histogram(data,xrange):
    deltax = 0.5*(xrange[1]-xrange[0])  # Anchura de cada bin
    N = len(data)
    binvalues = []
    binerrors = []
    for i in range(0,len(xcenters)):
        count = 0
        for j in data:
            if ((j > xcenters[i] - deltax) and (j <= xcenters[i] + deltax)):
                count += 1
        binvalues.append(count/(N*2*deltax))
        binerrors.append(np.sqrt(count/(2*deltax))/N)
    return ([binvalues,binerrors])

""" Cálculo de errores mediante promedios """
array_bin_values = [np.array(histogram(i,xrange)[0]) for i in data]
mean_bin_values = sum(array_bin_values)/len(array_bin_values)
mean_bin_values_square = sum([pow(i,2) for i in array_bin_values])/len(array_bin_values)
array_deviations = np.sqrt(mean_bin_values_square - pow(mean_bin_values,2))
def probexp(x,data,xrange):
    deltax = 0.1
    N = len(data)
    count = 0
    for i in data:
        if (abs(x-i) < deltax): 
            count += 1
    return count/N
    
def probfunction(x,mu,epsilon):
    omega = np.sqrt(pow(mu,2)*(1+0.25*pow(epsilon*mu,2)))
    return np.sqrt(omega/np.pi)*np.exp(-omega*x**2)


yvaluesexp ,yexperrors = totalhistogram(data,xrange)
yvaluestheo = [probfunction(i,parameters[3],parameters[1]) for i in xrange]
plt.plot(xrange, yvaluestheo, '--', color = 'blue', label = "Valores teóricos")
plt.errorbar(xcenters, mean_bin_values, yerr= array_deviations, linestyle = '-', color = 'red', label = "Valores simulación")
plt.ylabel(r'$|\psi_0(x)|^2$')
plt.xlabel("Positions x")
plt.title("Distribución probabilidad del estado fundamental, " + r'$\mu = $' + str(parameters[3]))
plt.legend(loc = "upper right")
plt.show()
