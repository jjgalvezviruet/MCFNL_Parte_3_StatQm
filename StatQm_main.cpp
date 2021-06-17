/* Implementación del procedimiento del artículo de Creutz y Freedman "A Statistical
Approach to QUantum Mechanics" */

#include "StatQm_header.h"

/* 1. Parámetros */
// Parámetros físicos
int main() {
float mu = sqrt(2);
float lambda = 0.0;
float M0 = 0.5;
float pi = 3.1415;
int N = 10000;
float epsilon = 0.5; //(2*pi)/(20*pow(mu,2));
float T = N*epsilon; 
cout << epsilon << endl;

// Parámetros Metrópolis
int Ne = 300;
int pMC = 5;
int n = 10;
long double delta = 2*sqrt(epsilon);

// Vector de posiciones 
long double positions[N]; 
int vecinos[N][2];

// Acción y potencial 
// S = epsilon* sum_i (0.5*M0* pow((x_i+1 - x_i),2)/ pow(epsilon,2)) + V(x_i)
// V = 0.5*pow(mu*x_i,2) + lambda*pow(x_i,4)


/* Generación de números aleatorios usando el algoritmo de Marsenne twister */

// Use random_device to generate a seed for Mersenne twister engine.
random_device rd{};    

// Use Mersenne twister engine to generate pseudo-random numbers.
mt19937 engine{rd()};

// "Filter" MT engine's output to generate pseudo-random double values,
// **uniformly distributed** on the closed interval [0, 1].
// (Note that the range is [inclusive, inclusive].)
uniform_real_distribution<long double> uniform01{0.0, 1.0};
float x = uniform01(engine); //Ejemplo de generación de número aleatorio entre 0 y 1

// Inicio de los vecinos
neighbors(&(vecinos[0][0]),N);

/* 2. Inicio vector posiciones */
for (int i = 0; i<N; i++)
{
    positions[i] = uniform01(engine)*6 -3;
} 

/* 3. Se genera un vector de posiciones distibuido según la distribución de 
probabilidad obtenida mediante el Hamiltoniano del sistema usando el método de 
metrópolis */

/* Escritura de datos */
ofstream file;
file.open("Results.csv",std::fstream::app);
file << N << "," << epsilon << "," << T << "," << mu << "," << lambda << "," << M0 << endl;

for (int p = 0; p < Ne; p++)
    {
    for (int k = 0; k<pMC; k++)
        {
        for (int i = 0; i<N; i++)
            {
                for(int j = 0; j<n; j++)
                {
                    long double oldpos = positions[i];

                    // Aspirante a nueva posicion 
                    long double newpos = oldpos + (uniform01(engine)*2*delta - delta); 

                    // Solo deltaS
                    long double deltaS = (M0/epsilon)*(pow(newpos,2) - pow(oldpos,2) - (newpos - oldpos)*
                                    (positions[vecinos[i][0]] + positions[vecinos[i][1]])) + 
                                    0.5*epsilon*pow(mu,2)*(pow(newpos,2) - pow(oldpos,2)) + 
                                    lambda*epsilon*(pow(newpos,4) - pow(oldpos,4));
                    
                    // Algoritmo aceptación/rechazo
                    if (deltaS < 0) {positions[i] = newpos;}
                    else
                    {
                        long double r = uniform01(engine);
                        if (exp(-deltaS) > r) {positions[i] = newpos;}
                        else {positions[i] = oldpos;}
                    }
                }
            }
        }

        for (int i = 0; i<N; i++)
        {
            file << positions[i] << ",";
        }
        file << endl;
    }
file.close();
}

