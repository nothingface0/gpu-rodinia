

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"





__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   
   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads();
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();   
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }
   
   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}

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
//	    printf ("correcting tid=%d %.10f %.10f %.10f\n", tid,result[tid], result[tid+size], result[tid+size*2]);  
	    result[tid] = result[tid+size*2];
	}
    }   
}


__global__ void prepare_dup(float* a, int size)
{
	
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;
    a[tid+size] = a[tid];
    a[tid+size*2] = a[tid];
}

/* END of Lishan add */




__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int cur_copy = blockIdx.x;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index+cur_copy*139281] = w[index];
   __syncthreads();
   w[index+cur_copy*139281] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index+cur_copy*139281] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x+cur_copy*139281] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x+cur_copy*139281]));
   oldw[index_x+cur_copy*139281] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x+cur_copy*139281]));
   }

}
#endif 
