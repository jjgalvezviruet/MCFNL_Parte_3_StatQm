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

data = []
parameters = []
for j in range(7,13):
    auxparam = []
    auxdata = []
    with open('Energyanharmonicdata/Results' + str(j) + ".csv") as file2:
        csv_reader = csv.reader(file2, delimiter=',')
        line_count = 0
        for i in csv_reader:
            if (line_count == 0):
                auxparam = list(map(lambda x: float(x),i))
                line_count += 1
            else:
                auxdata.append(np.array(list(map(lambda x: float(x),i[:-1]))))
                line_count += 1
    data.append(auxdata)
    parameters.append(auxparam)
                 
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

def correlationfunction(data,n,N):
    correlations = []
    for i in range(0,N):
        auxcorr = []
        for j in data:
            auxcorr.append(arraycorr(j,i*n))
        correlations.append(auxcorr)
    return correlations

def Energy0 (data,parameters,Ne,datagroup = 25):
    Ne = len(data)
    ne = int(Ne/datagroup)    
    mediaposfourth = divide(pow(np.array(data).transpose(),4),ne,Ne,positive = False)
    mediapossquare = divide(pow(np.array(data).transpose(),2),ne,Ne,positive = False)
    Energias = -4*pow(parameters[6],2) * np.array([np.mean(i) for i in np.array(mediapossquare).transpose()]) + \
        3 * parameters[4] * np.array([np.mean(i) for i in np.array(mediaposfourth).transpose()]) + \
        parameters[4]*pow(parameters[6],4)

    meanEnergias = np.mean(Energias)
    stdEnergias = np.sqrt(sum(pow(Energias-meanEnergias,2))/(len(Energias)-1))
    return [meanEnergias, stdEnergias]


def Energydifference(data,parameters, n = 1, N = 6, nvalues = 3, n0 = 0, datagroup = 25):
    correlations = []  # Array de correlaciones de cada uno de los datos
    Ne = len(data)
    ne = int(Ne/datagroup)
    """ Cálculo de correlaciones """
    
    """ 1. Calculamos las correlaciones """
    correlations = correlationfunction(data,n,N)
    
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

    return [[pendientefinal,desviacionpendiente,fluctuacionespendiente], \
            [ordenadafinal,desviacionordenada,fluctuacionesordenada]]

# Parámetros
n = 2             # Salto entre correlaciones consecutivas
N = 10             # N-1, numero de correlaciones calculadas
nvalues = 5       # Nº de datos para ajuste
n0 = 0            # Punto inicial en el que empezar el cálculo
datagroup = 5     # Nº de datos en cada grupo
Ne = len(data[0])
ne = int(Ne/datagroup)



""" Representacon datos """
# Array tiempos
times = [i*parameters[0][1]*n for i in range(0,N)]

arraydif = []
arrayE0 = []

for j in range(0,6):
    # Resultados Enegia
    Results = Energydifference(data[j],parameters[j],n,N,nvalues,n0,datagroup)
    arraydif.append(Results)
    resultsE0 = Energy0(data[j],parameters[j],Ne,5)
    arrayE0.append(resultsE0)

yE0 = [i[0] for i in arrayE0]
yE1 = [arrayE0[i][0]-arraydif[i][0][0] for i in range(0,len(arrayE0))]
arrayfs = np.array([pow(i[6],2) for i in parameters])


plt.plot(arrayfs, yE0, ".", label = "Energías E0")
plt.plot(arrayfs, yE1, ".", label = "Energías E1")
plt.xlim(arrayfs[0]+0.5,arrayfs[-1]-0.5)
plt.legend(loc = "lower left")
plt.title("Energías E1 y E0 para distintos valores de f")
plt.ylabel(r'$E(f^2)$')
plt.xlabel(r'$f^2$')


