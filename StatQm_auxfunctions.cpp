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