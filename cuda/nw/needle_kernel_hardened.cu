
#include "needle.h"
#include <stdio.h>


#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

__device__ __host__ int 
maximum( int a,
		 int b,
		 int c){

int k;
if( a <= b )
k = b;
else 
k = a;

if( k <=c )
return(c);
else
return(k);

}

__global__ void
needle_cuda_shared_1(  int* referrence,
			  int* matrix_cuda, 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
  
  __shared__  int temp[BLOCK_SIZE+1][(BLOCK_SIZE+1)*3]; // modified by Lishan
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE*3]; // modified by Lishan 

  // __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1]; // original code
  // __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE]; // original code 
 
  if (tx == 0)
    temp[tx][0+threadIdx.y*(BLOCK_SIZE+1)] = matrix_cuda[index_nw]; // modified by Lishan 
  								// cur_copy=threadIdx.y
    // temp[tx][0] = matrix_cuda[index_nw]; // original code


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
    ref[ty][tx+threadIdx.y*BLOCK_SIZE] = referrence[index + cols * ty]; // modified by Lishan
    // ref[ty][tx] = referrence[index + cols * ty]; // original code

  __syncthreads();

  temp[tx + 1][0+threadIdx.y*(BLOCK_SIZE+1)] = matrix_cuda[index_w + cols * tx]; // modified by Lishan
  // temp[tx + 1][0] = matrix_cuda[index_w + cols * tx]; // original code

  __syncthreads();

  temp[0][tx + 1 + threadIdx.y*(BLOCK_SIZE+1)] = matrix_cuda[index_n]; // modified by Lishan
  // temp[0][tx + 1] = matrix_cuda[index_n]; // original code
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          /* START of Lishan code */
          temp[t_index_y][t_index_x+threadIdx.y*(BLOCK_SIZE+1)] = maximum( temp[t_index_y-1][t_index_x-1+threadIdx.y*(BLOCK_SIZE+1)] + ref[t_index_y-1][t_index_x-1+threadIdx.y*BLOCK_SIZE],
		                                        temp[t_index_y][t_index_x-1+threadIdx.y*(BLOCK_SIZE+1)]  - penalty, 
				
          /* END of Lishan code */							temp[t_index_y-1][t_index_x+threadIdx.y*(BLOCK_SIZE+1)]  - penalty);

	  /* START of original code 
	  temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);
	  END of original code */
	  
	  
	  }

	  __syncthreads();
  
    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          /* START of Lishan code */
          temp[t_index_y][t_index_x+threadIdx.y*(BLOCK_SIZE+1)] = maximum( temp[t_index_y-1][t_index_x-1+threadIdx.y*(BLOCK_SIZE+1)] + ref[t_index_y-1][t_index_x-1+threadIdx.y*BLOCK_SIZE],
		                                        temp[t_index_y][t_index_x-1+threadIdx.y*(BLOCK_SIZE+1)]  - penalty, 
												temp[t_index_y-1][t_index_x+threadIdx.y*(BLOCK_SIZE+1)]  - penalty);
	  /* END of Lishan code */

          /* START of original code
          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);
	   END of original code */
	  }

	  __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  {
  /* START of Lishan adding */
  if (threadIdx.y == 0)
	if (temp[ty+1][tx+1] != temp[ty+1][tx+1+BLOCK_SIZE+1])
		temp[ty+1][tx+1] = temp[ty+1][tx+1+(BLOCK_SIZE+1)*2];

  /* END of Lishan adding */
  	matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];
  }

}


__global__ void
needle_cuda_shared_2(  int* referrence,
			  int* matrix_cuda, 
			 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i  ;
  int b_index_y = block_width - bx -1;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
    int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__  int temp[BLOCK_SIZE+1][(BLOCK_SIZE+1)*3]; // L
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE*3]; // L

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx+threadIdx.y * BLOCK_SIZE] = referrence[index + cols * ty]; // L

  __syncthreads();

   if (tx == 0)
		  temp[tx][0+(BLOCK_SIZE+1)*threadIdx.y] = matrix_cuda[index_nw]; // L
 
 
  temp[tx + 1][0+(BLOCK_SIZE+1)*threadIdx.y] = matrix_cuda[index_w + cols * tx]; //L

  __syncthreads();

  temp[0][tx + 1+(BLOCK_SIZE+1)*threadIdx.y] = matrix_cuda[index_n]; //L
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x+(BLOCK_SIZE+1)*threadIdx.y] = maximum( temp[t_index_y-1][t_index_x-1+(BLOCK_SIZE+1)*threadIdx.y] + ref[t_index_y-1][t_index_x-1+BLOCK_SIZE*threadIdx.y],
		                                        temp[t_index_y][t_index_x-1+(BLOCK_SIZE+1)*threadIdx.y]  - penalty, 
												temp[t_index_y-1][t_index_x+(BLOCK_SIZE+1)*threadIdx.y]  - penalty);	 // L 
	  
	  }

	  __syncthreads();
  
    }


 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x+(BLOCK_SIZE+1)*threadIdx.y] = maximum( temp[t_index_y-1][t_index_x-1+(BLOCK_SIZE+1)*threadIdx.y] + ref[t_index_y-1][t_index_x-1+BLOCK_SIZE*threadIdx.y],
		                                        temp[t_index_y][t_index_x-1+(BLOCK_SIZE+1)*threadIdx.y]  - penalty, 
												temp[t_index_y-1][t_index_x+(BLOCK_SIZE+1)*threadIdx.y]  - penalty);


	  }

	  __syncthreads();
  }


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  {
    if (threadIdx.y == 0)
    {
      if (temp[ty+1][tx+1] != temp[ty+1][tx+1+(BLOCK_SIZE+1)])
	      temp[ty+1][tx+1] = temp[ty+1][tx+1+(BLOCK_SIZE+1)*2];
      matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];
    }
  }

}

