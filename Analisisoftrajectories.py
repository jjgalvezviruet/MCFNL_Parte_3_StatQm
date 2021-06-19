#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:15:47 2021

@author: juanjo
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv

data = [[],[],[]]
parameters = [[],[],[]]
for j in range(1,4):
    with open('Datatrajectories/Results' + str(j) + ".csv") as file2:
        csv_reader = csv.reader(file2, delimiter=',')
        line_count = 0
        for i in csv_reader:
            if (line_count == 0):
                parameters[j-1] = list(map(lambda x: float(x),i))
                line_count += 1
            else:
                data[j-1].append(np.array(list(map(lambda x: float(x),i[:-1]))))
                line_count += 1
                
tiempos = [parameters[0][1]*i for i in range(0,int(parameters[0][0]))]
xrange = np.linspace(-3,3,100)

mu0 = parameters[0][3]
mu1 = parameters[1][3]
mu2 = parameters[2][3]

plt.plot(data[0][0][0:1000],tiempos[0:1000], linewidth = 0.6, \
         linestyle = "--", color = "blue", label = r'$\mu$ = ' +  str(mu0))
plt.plot(data[1][0][0:1000],tiempos[0:1000], linewidth = 0.6, \
         linestyle = "-.", color = "red", label = r'$\mu$ = ' + str(mu1))
plt.plot(data[2][0][0:1000],tiempos[0:1000], linewidth = 0.6, \
         color = "green", label = r'$\mu$ = ' + str(mu2))
plt.xlabel("Position x")
plt.ylabel("Time " + r'$\tau$')
plt.title("Possible trajectories for harmonic oscillators")
plt.legend(loc = "upper right")
plt.show()