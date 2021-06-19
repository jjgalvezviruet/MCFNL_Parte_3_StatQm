#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:30:45 2021

@author: juanjo
"""

import numpy as np 
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression

with open('Energydata/Results23.csv') as file2:
    csv_reader = csv.reader(file2, delimiter=',')
    line_count = 0
    data = []
    for i in csv_reader:
        if (line_count == 0):
            parameters = list(map(lambda x: np.float128(x),i))
            line_count += 1
        else:
            data.append(np.array(list(map(lambda x: np.float128(x),i[:-1]))))
            line_count += 1
            
def movearray(array,n):
    N = len(array)
    arrayaux = [array[i+n] for i in range(0,N) if i+n < N]
    for i in range(0,n):
        arrayaux.append(array[i])
    return np.array(arrayaux)

def arraycorr(array,n):
    arrayaux = movearray(array,n)
    return np.mean(arrayaux*array)

""" Error de coeficiente de regresion """
def errorregression(coefs,yvalues,xvalues):
    """ Error pendiente """
    nvalues = len(xvalues)
    pendiente = coefs[0]
    ordenada = coefs[1]
    
    # Calculo de desviacion de datos respecto a predicción
    yprediccion = np.array([i*pendiente+ordenada for i in xvalues])
    predictiovariance = sum(pow(yvalues-yprediccion,2))/(nvalues-2)
    
    # Calculo de desviacion de pendiente
    meanx = np.mean(xvalues)
    xdeviations = sum(pow((xvalues-meanx),2))
    pendienteerror = np.sqrt(predictiovariance/xdeviations)
    
    # Error ordenada
    ordenadaerror = pendienteerror*np.sqrt(sum(pow(xvalues,2))/nvalues)
    
    return [ordenadaerror,pendienteerror]

def prediccion_fit(coefs,xvalues):
    pendiente = coefs[0]
    ordenada = coefs[1]
    return np.array([np.exp(i*pendiente+ordenada) for i in xvalues])


def divide(data,n,N,positive = True):
    result = []
    ndataingroups = int(N/n)
    for k in data:
        auxarray = []
        for i in range(0,n):
            aux = np.mean([j for j in k[ndataingroups*i:ndataingroups*(i+1)]])
            if positive: 
                if aux > 0: auxarray.append(aux)
            else: auxarray.append(aux)
        result.append(auxarray)
    return result


n = 1             # Salto entre correlaciones consecutivas
N = 6             # N-1, numero de correlaciones calculadas
nvalues = 3       # Nº de datos para ajuste
n0 = 0            # Punto inicial en el que empezar el cálculo
correlations = []  # Array de correlaciones de cada uno de los datos
Ne = len(data)
ne = int(Ne/25)
""" Cálculo de correlaciones """

""" 1. Calculamos las correlaciones """
for i in range(0,N):
    auxcorr = []
    for j in data:
        auxcorr.append(arraycorr(j,i*n))
    correlations.append(auxcorr)

""" 2. Dividimos las correlaciones en ne grupos """
correlationsgroup = divide(correlations[n0:nvalues],ne,Ne)
# Nos quedamos solo ocn el número minimo de columnas
correlationsgroup = [j[0:min([len(i) for i in correlationsgroup])] for j in correlationsgroup]

Results = []
for i in np.array(correlationsgroup).transpose():
    meancorrelations = i # media de correlaciones de todos los datos
    nvariables = len(meancorrelations)
    
    times = [i*parameters[1]*n for i in range(0,N)]

    """ Ajuste lineal de correlaciones """
    logcorrelations = np.array([np.log(i) for i in meancorrelations]).reshape(-1,1) #logaritmo correlaciones
    regtimes = np.array([[i] for i in times[n0:nvariables]])  # Tiempos usados para regresion
    reg = LinearRegression().fit(regtimes,logcorrelations)  # Regresión lineal

    """Calculo de errores """
    [ordenadaerror,pendienteerror] = errorregression([reg.coef_[0][0],reg.intercept_[0]],np.log(meancorrelations[n0:nvariables]),np.array(times[n0:nvariables]))

    Results.append([[reg.coef_[0][0],pendienteerror],[reg.intercept_[0],ordenadaerror]])

pendientes = np.array([i[0][0] for i in Results])
ordenadas = np.array([np.exp(i[1][0]) for i in Results])

pendientefinal = np.mean(pendientes)
ordenadafinal = np.mean(ordenadas)

desviacionpendiente = np.sqrt(sum(pow(pendientes-pendientefinal,2))/(len(pendientes)-1))
desviacionordenada = np.sqrt(sum(pow(ordenadas-ordenadafinal,2))/(len(ordenadas)-1))


fluctuacionespendiente = desviacionpendiente/abs(pendientefinal)
fluctuacionesordenada = desviacionordenada/abs(ordenadafinal)

print ("E1 - E0 = " + str(-pendientefinal) + ", Desviación = " + str(desviacionpendiente) + ", Fluctuaciones = " + str(fluctuacionespendiente))
print ("<x^2> = " + str(ordenadafinal) + ", Desviación = " + str(desviacionordenada) + ", Fluctuaciones = " + str(fluctuacionesordenada))



""" Representacon datos """
meancorrelationstotal = np.array([np.mean(i) for i in correlations])
plt.semilogy(times[0:N],  meancorrelationstotal,'.',label = "Valores correlación")
labelajuste = "Ajuste " + str(nvalues) + " primeras correlaciones"
title = "Ajuste correlaciones para N = " + str(len(data[0])) + " y Ne = " + str(Ne)
plt.semilogy(times[0:N], prediccion_fit([pendientefinal,np.log(ordenadafinal)],times[0:N]),linestyle = '--', color = 'green', label = labelajuste)
plt.title(title)
plt.legend(loc = "upper right")
plt.xlabel(r'$\tau$')
plt.ylabel(r'$<x(0)x(\tau)>$')

plt.grid(True, which="both", ls="-")
plt.show()


""" Cálculo de E0 """
"""1. Dividimos la energía en ne grupos """

mediaposfourth = divide(pow(np.array(data).transpose(),4),ne,Ne,positive = False)
mediapossquare = divide(pow(np.array(data).transpose(),2),ne,Ne,positive = False)
Energias = pow(parameters[3],2) * np.array([np.mean(i) for i in np.array(mediapossquare).transpose()]) + \
3 * parameters[4] * np.array([np.mean(i) for i in np.array(mediaposfourth).transpose()])

meanEnergias = np.mean(Energias)
stdEnergias = np.sqrt(sum(pow(Energias-meanEnergias,2))/(len(Energias)-1))

print("Energía E0 = " + str(meanEnergias) + ", Desviación = " + str(stdEnergias))