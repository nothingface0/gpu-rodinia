#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

void
init(int argc, char** argv)
{
	if(argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]); 
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

/* START of Lishan add */
__global__ void check_correctness(int* result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (result[tid] != result[tid+10000])
    {
	if (result[tid] != result[tid+10000*2] && result[tid+10000]!= result[tid+10000*2])
	{ 
	    printf ("DUE %d %d %d\n", result[tid], result[tid+10000], result[tid+10000*2]);  
	    // All three copies have different results. This is considered as DUE, not SDC.
	}
	else
	{
	    // printf ("correcting tid=%d %.10f %.10f %.10f\n", tid,result[tid], result[tid+65536], result[tid+65536*2]);  
	    result[tid] = result[tid+10000*2];
	}
    }   
}
/* END of Lishan add */

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{
	/* START of Lishan add */
        __shared__ int prev[BLOCK_SIZE*3]; 
        __shared__ int result[BLOCK_SIZE*3];  

	int bx = blockIdx.x;
	int tx=threadIdx.x%BLOCK_SIZE; 
	int cur_copy = threadIdx.x / BLOCK_SIZE; 
	/* END of Lishan add */

	/* START of original code	
        __shared__ int prev[BLOCK_SIZE];  
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x; 
	END of original code */
	       
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[threadIdx.x] = gpuSrc[xidx]; // modified by Lishan
//            prev[tx] = gpuSrc[xidx]; // original code
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
			
		  /* START of Lishan add */
                  int left = prev[W+BLOCK_SIZE*cur_copy]; 
                  int up = prev[tx+BLOCK_SIZE*cur_copy]; 
                  int right = prev[E+BLOCK_SIZE*cur_copy]; 
		  /* END of Lishan add */
 
		  /* START of original code
                  int left = prev[W]; 
                  int up = prev[tx]; 
                  int right = prev[E]; 
		    END of original code */

                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[threadIdx.x] = shortest + gpuWall[index]; // modified by Lishan
	        //  result[tx] = shortest + gpuWall[index]; // original code

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[threadIdx.x]= result[threadIdx.x]; // modified by Lishan
		// prev[tx]= result[tx]; // original code
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
	/* START of Lishan add, this part is for checking code in the same kernel *
	if (cur_copy == 0)
	{
	  if (result[tx] != result[tx+BLOCK_SIZE]) // copy 0 != copy 1
	  {
	    gpuResults[xidx]=result[tx+BLOCK_SIZE*2];			 
	  }
	  else
	  {
	    gpuResults[xidx]=result[tx]; 
	  }
	}
	* END of Lishan add */

	gpuResults[xidx + 10000*cur_copy] = result[tx+BLOCK_SIZE*cur_copy]; // modified by Lishan
	// cols=10000, hard-coded
	
	//  gpuResults[xidx]=result[tx]; // original code
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE*3); // Modified by Lishan, thds per block * 3
	// dim3 dimBlock(BLOCK_SIZE); // original code
        dim3 dimGrid(blockCols);  
	dim3 dimBlockHardening(256); // added by Lishan
	dim3 dimGridHardening(10000/256); // added by Lishan


        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
    	    check_correctness<<<dimGridHardening, dimBlockHardening>>>(gpuResult[dst]);	// added by Lishan

            // for the measurement fairness
            cudaDeviceSynchronize();
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols*3); //L
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols*3); //L
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
}

