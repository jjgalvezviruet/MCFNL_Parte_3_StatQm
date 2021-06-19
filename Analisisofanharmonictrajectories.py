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

data = []
parameters = []
with open("Datatrajectoriesanharmonic/Results3.csv") as file2:
    csv_reader = csv.reader(file2, delimiter=',')
    line_count = 0
    for i in csv_reader:
        if (line_count == 0):
            parameters = list(map(lambda x: float(x),i))
            line_count += 1
        else:
            data.append(np.array(list(map(lambda x: float(x),i[:-1]))))
            line_count += 1
                
tiempos = [parameters[1]*i for i in range(0,int(parameters[0]))]
xrange = np.linspace(-3,3,100)

f = parameters[6]
f2 = 0.5

plt.plot(data[0][0:150],tiempos[0:150], linewidth = 0.8, \
         linestyle = "-", color = "blue", label = r'$f^2$ = ' +  str(f2))
plt.axline([f,data[0][0]],[f,data[0][150]], color = "green" ,linestyle = "--",\
           linewidth = 0.8)
plt.axline([-f,data[0][0]],[-f,data[0][150]], color = "green" ,linestyle = "--",\
           linewidth = 0.8)
plt.xlabel("Position x")
plt.ylabel("Time " + r'$\tau$')
plt.title("Possible trajectory for anharmonic oscillator")
plt.legend(loc = "upper right")
plt.show()