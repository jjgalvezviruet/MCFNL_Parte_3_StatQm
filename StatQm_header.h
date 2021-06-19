/* Header del proyecto */
#ifndef ISING_HEADER_H 
#define ISING_HEADER_HS

#include <random>
#include <math.h>  
#include <iostream>
#include <fstream>
using namespace std;

// Auxiliar functions
void neighbors (int *p, int size);
long double deltaV (long double xnew, long double xold, float mu, float lambda, float epsilon, float f, int type);

#endif /* ISING_HEADER_H */