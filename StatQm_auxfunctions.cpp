#include "StatQm_header.h"

void neighbors (int *p, int size)
{
    for (int i = 0; i<size; i++)
    {
        if (i == 0)
        {
            *p = size-1;
            *(p + 1) = +1;
        }
        else if(i == size-1)
        {
            *(p + i*2) = i - 1;
            *(p + i*2 + 1) = 0;
        }
        else
        {
            *(p + i*2) = i -1;
            *(p + i*2 + 1) = i +1;
        }

    }
}

long double deltaV (long double xnew, long double xold, float mu, float lambda, float epsilon, float f = 1, int type = 1)
{
    switch(type){
        case 1: return 0.5*epsilon*pow(mu,2)*(pow(xnew,2) - pow(xold,2)) + 
                                    lambda*epsilon*(pow(xnew,4) - pow(xold,4));
        case 2: return lambda*epsilon*(pow(xnew,4)-pow(xold,4)) - 2*epsilon*pow(f,2)*
                                    (pow(xnew,2)-pow(xold,2));
    }
}