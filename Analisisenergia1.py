#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:42:45 2021

@author: juanjo
"""
import numpy as np 
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression

with open('Energydata/Results9.csv') as file2:
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
    predictiovariance = sum(pow(yvalues-yprediccion,2))/(nvalues-1)
    
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

def all_positive(array):
    for i in array:
        if i < 0: return False
    return True

        
n = 1             # Salto entre correlaciones consecutivas
N = 6              # N-1, numero de correlaciones calculadas
nvalues = 3        # Nº de datos para ajuste
correlations = []  # Array de correlaciones de cada uno de los datos

""" Cálculo de correlaciones """
for i in range(0,N):
    auxcorr = []
    for j in data:
        auxcorr.append(arraycorr(j,i*n))
    correlations.append(auxcorr)


positivecorr = np.array(list(filter(all_positive,np.array(correlations[0:nvalues]).transpose())))
results = []
for k in positivecorr:
    times = [i*parameters[1]*n for i in range(0,N)]
    
    """ Ajuste lineal de correlaciones """
    logcorrelations = np.array([np.log(i) for i in k[0:nvalues]]).reshape(-1,1) #logaritmo correlaciones
    regtimes = np.array([[i] for i in times[0:nvalues]])  # Tiempos usados para regresion
    reg = LinearRegression().fit(regtimes,logcorrelations)  # Regresión lineal
    
    """Calculo de errores """
    [ordenadaerror,pendienteerror] = errorregression([reg.coef_[0][0],reg.intercept_[0]],np.log(k[0:nvalues]),np.array(times[0:nvalues]))
    
    
    results.append([[reg.coef_[0][0],pendienteerror],[reg.intercept_[0],ordenadaerror]])

valuespendiente = [i[0][0] for i in results]
mediapendiente = np.mean([i[0][0] for i in results])

valeusordenada = [i[1][0] for i in results]
mediaordenada = np.mean([i[1][0] for i in results])

stddeviation = np.sqrt(sum(pow(valuespendiente-mediapendiente,2))/(len(valuespendiente)-1))
fluctuations = stddeviation/abs(mediapendiente)

print("E1-E0 = " + str(-mediapendiente) + ", Fluctuaciones = " + str(fluctuations))

""" Representación gráfica """
meancorrelations = [np.mean(i) for i in correlations]

plt.semilogy(times,  meancorrelations,'.')
plt.semilogy(times, prediccion_fit([mediapendiente,mediaordenada],times),linestyle = '--', color = 'green')
plt.show()



