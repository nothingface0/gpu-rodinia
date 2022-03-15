#include <stdio.h>
/* START of Lishan add */
__global__ void check_correctness(float* result, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size) return;
	
    if (result[tid] != result[tid+size])
    {
	if (result[tid] != result[tid+size*2] && result[tid+size]!= result[tid+size*2])
	{ 
	    printf ("DUE %d %d %d\n", result[tid], result[tid+size], result[tid+size*2]);  
	    // All three copies have different results. This is considered as DUE, not SDC.
	}
	else
	{
	    //printf ("correcting tid=%d %.10f %.10f %.10f\n", tid,result[tid], result[tid+size], result[tid+size*2]);  
	    result[tid] = result[tid+size*2];
	}
    }   
}
/* END of Lishan add */





