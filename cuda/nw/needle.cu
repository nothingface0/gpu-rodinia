#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "needle_kernel.cu"

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

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
  const char *results = "print traceback value GPU:\n\
2 12 6 8 18 12 15 7 7 2 -7 -8 2 -7 -9 -9 1 -5 -13 -13 -22 -12 -18 -15 -12 -20 -20 -18 -20 -16 -14 -11 -8 -13 -12 -20 -20 -16 -16 -12 -8 -4 -6 -4 -13 -13 -13 -3 -8 -14 -23 -27 -26 -26 -23 -19 -18 -27 -17 -25 -24 -33 -37 -27 -17 -23 -21 -23 -27 -24 -21 -23 -20 -20 -18 -14 -19 -16 -13 -12 -9 -15 -13 -22 -22 -22 -22 -19 -17 -17 -13 -13 -21 -20 -20 -20 -25 -25 -22 -19 -21 -25 -34 -43 -40 -42 -40 -36 -44 -42 -43 -43 -43 -40 -38 -37 -34 -39 -36 -37 -43 -43 -48 -48 -47 -45 -50 -49 -46 -43 -41 -38 -35 -33 -41 -40 -30 -28 -34 -31 -31 -40 -37 -33 -39 -38 -37 -35 -36 -42 -42 -42 -40 -37 -37 -37 -34 -31 -33 -29 -38 -43 -43 -48 -50 -47 -46 -46 -46 -50 -48 -46 -51 -51 -49 -47 -51 -51 -48 -50 -50 -46 -48 -44 -46 -45 -54 -51 -47 -49 -48 -45 -44 -45 -45 -41 -38 -43 -43 -42 -41 -31 -32 -29 -26 -26 -31 -28 -24 -23 -20 -26 -24 -29 -19 -28 -30 -30 -28 -27 -26 -26 -25 -33 -30 -39 -29 -31 -22 -22 -21 -21 -17 -18 -15 -12 -16 -13 -17 -17 -15 -11 -11 -11 -16 -13 -11 -11 -20 -26 -28 -28 -26 -26 -30 -30 -34 -34 -34 -31 -21 -23 -31 -31 -30 -28 -28 -26 -25 -25 -33 -34 -34 -39 -35 -33 -37 -34 -35 -33 -30 -29 -33 -23 -28 -24 -14 -14 -19 -9 -7 -15 -11 -8 -13 -19 -9 1 0 ";

  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *referrence;
	int *matrix_cuda,  *referrence_cuda;
	int size;
    
    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
	usage(argc, argv);
    }
	
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}
	

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );
	
	
    for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}
	
	printf("Start Needleman-Wunsch\n");
	
	for( int i=1; i< max_rows ; i++){    //please define your own sequence. 
       input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
    for( int j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;


    size = max_cols * max_rows;
	cudaMalloc((void**)& referrence_cuda, sizeof(int)*size);
	cudaMalloc((void**)& matrix_cuda, sizeof(int)*size);
	
	cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);

    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

#ifdef  TIMING
  gettimeofday(&tv_kernel_start, NULL);
#endif

	printf("Processing top-left matrix\n");
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda
		                                      ,max_cols, penalty, i, block_width); 
	}
	printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
		needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda
		                                      ,max_cols, penalty, i, block_width); 
	}

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);

#define TRACEBACK
#ifdef TRACEBACK

	// FILE *fpo = fopen("result.txt","w");
	// fprintf(fpo, "print traceback value GPU:\n");
  char *str_result = new char[(sizeof(char) * (strlen(results)+1))];
  str_result[0] = 0;

  sprintf(str_result + strlen(str_result), "print traceback value GPU:\n");

	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			// fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
      sprintf(str_result + strlen(str_result), "%d ", output_itemsets[ i * max_cols + j]);
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
		    w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = output_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
		// fprintf(fpo, "%d ", traceback);
    sprintf(str_result + strlen(str_result), "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}

  if (strcmp(results, str_result) == 0) {
      printf("Test PASSED\n");
    } else {
      printf("Test FAILED\n");
    }

	// fclose(fpo);
  delete str_result;

#endif

	cudaFree(referrence_cuda);
	cudaFree(matrix_cuda);

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

#ifdef  TIMING
    printf("Exec: %f\n", kernel_time);
#endif
}
