#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:45:03 2021

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


def ajustelinealdatos(n = 1,N = 6, nvalues = 3,data = data):
    n              # Salto entre correlaciones consecutivas
    N              # N-1, numero de correlaciones calculadas
    nvalues        # Nº de datos para ajuste
    correlations = []  # Array de correlaciones de cada uno de los datos
    
    """ Cálculo de correlaciones """
    for i in range(0,N):
        auxcorr = []
        for j in data:
            auxcorr.append(arraycorr(j,i*n))
        correlations.append(auxcorr)
    
    meancorrelations = [np.mean(i) for i in correlations] # media de correlaciones de todos los datos
    times = [i*parameters[1]*n for i in range(0,N)]
    
    """ Ajuste lineal de correlaciones """
    logcorrelations = np.array([np.log(i) for i in meancorrelations[0:nvalues]]).reshape(-1,1) #logaritmo correlaciones
    regtimes = np.array([[i] for i in times[0:nvalues]])  # Tiempos usados para regresion
    reg = LinearRegression().fit(regtimes,logcorrelations)  # Regresión lineal
    
    """Calculo de errores """
    [ordenadaerror,pendienteerror] = errorregression([reg.coef_[0][0],reg.intercept_[0]],np.log(meancorrelations[0:nvalues]),np.array(times[0:nvalues]))


    """ Representacon datos """
    plt.semilogy(times,  meancorrelations,'.')
    plt.semilogy(times, prediccion_fit([reg.coef_[0][0],reg.intercept_[0]],times),linestyle = '--', color = 'green')
    plt.show()
    
    return [[reg.coef_[0][0],pendienteerror],[reg.intercept_[0],ordenadaerror]]


Results = ajustelinealdatos()
print("Pendiente: " + str(Results[0][0]) + ", Error: " + str(Results[0][1]))
print("Ordenada: " + str(Results[1][0]) + ", Error: " + str(Results[1][1]))






