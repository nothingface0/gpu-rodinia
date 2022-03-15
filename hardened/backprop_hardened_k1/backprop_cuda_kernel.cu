

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda, // Lishan's Q: this is not used.
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x % 16; // L

 //  if (tx >= 16) return;

   int cur_copy = threadIdx.x / 16; // aL

   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT*3]; //  L
   __shared__ float weight_matrix[HEIGHT][WIDTH*3]; // L

   // L
   if ( tx == 0 )
   input_node[ty+cur_copy*HEIGHT] = input_cuda[index_in] ;
   
   /* ori   
   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   */
   __syncthreads();

   weight_matrix[ty][tx+cur_copy *WIDTH] = input_hidden_cuda[index]; // L

   __syncthreads();
   
   weight_matrix[ty][tx+cur_copy*WIDTH] = weight_matrix[ty][tx+cur_copy*WIDTH] * input_node[ty+cur_copy*HEIGHT];

   __syncthreads();   
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx+cur_copy*WIDTH] = weight_matrix[ty][tx+cur_copy*WIDTH] + weight_matrix[ty + power_two/2][tx+cur_copy*WIDTH];

	   __syncthreads();

   }
   
   //__syncthreads();

   input_hidden_cuda[index+cur_copy*139281] = weight_matrix[ty][tx+cur_copy*WIDTH]; // L
   // input_hidden_cuda[index+139281] = weight_matrix[ty][tx+cur_copy*WIDTH]; // L
    // input_hidden_cuda[index+2*139281] = weight_matrix[ty][tx+cur_copy*WIDTH]; // L
  
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
	   hidden_partial_sum[by * hid + ty+cur_copy*8192] = weight_matrix[tx][ty+cur_copy*WIDTH]; // L
//	printf ("tx=%d, ty=%d, by=%d, hid=%d, cp=%d, hidden_partial_sum=%f\n", tx, ty, by, hid, cur_copy, hidden_partial_sum[by*hid+ty+cur_copy*8192]);
	   //hidden_partial_sum[by * hid + ty+8192] = weight_matrix[tx][ty+cur_copy*WIDTH]; // L
	   //hidden_partial_sum[by * hid + ty+2*8192] = weight_matrix[tx][ty+cur_copy*WIDTH]; // L
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
	    //printf ("correcting tid=%d %.10f %.10f %.10f\n", tid,result[tid], result[tid+size], result[tid+size*2]);  
	    result[tid] = result[tid+size*2];
	}
    }   
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
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
