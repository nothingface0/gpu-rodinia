

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
  BPNN *net, *net_cpu;
  int i;
  float out_err, hid_err, out_err_cpu, hid_err_cpu;
  int n1, n2, n3, j, k;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);

  srand(7);
  net_cpu = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  load(net_cpu);
  bpnn_train_cuda_CPU(net_cpu, &out_err_cpu, &hid_err_cpu);

  int error=0;
  //if(fabs(out_err - out_err_cpu) >= 0.01f) {
  // error=1;
  //}
  if(fabs(hid_err - hid_err_cpu) >= 0.01f) {
    error=1;
  }
  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  for (j = 0; j <= n1; j++) {
    for (k = 0; k <= n2; k++) {
      if(fabs((net->input_weights)[j][k] - (net_cpu->input_weights)[j][k]) >= 0.01f) {
	error=1;
      }
      if(fabs((net->input_prev_weights)[j][k] - (net_cpu->input_prev_weights)[j][k]) >= 0.01f) {
	error=1;
      }
    }
  }
  for (j = 0; j <= n2; j++) {
    for (k = 0; k <= n3; k++) {
      if(fabs((net->hidden_weights)[j][k] - (net_cpu->hidden_weights)[j][k]) >= 0.01f) {
	error=1;
      }
      if(fabs((net->hidden_prev_weights)[j][k] - (net_cpu->hidden_prev_weights)[j][k]) >= 0.01f) {
	error=1;
      }
    }
  }

  // for (j = 1; j <= n1; j++) {
  //   if(fabs((net->input_units)[j] - (net_cpu->input_units)[j]) >= 0.01f) {
  //     error=1;
  //   }
  // }

  for (j = 0; j <= n2; j++) {
    if(fabs((net->hidden_units)[j] - (net_cpu->hidden_units)[j]) >= 0.01f) {
	error=1;
    }
    //if(fabs((net->hidden_delta)[j] - (net_cpu->hidden_delta)[j]) >= 0.01f) {
	//error=1;
    //}
  }

  for (j = 0; j <= n3; j++) {
    // if(fabs((net->target)[j] - (net_cpu->target)[j]) >= 0.01f) {
    //   error=1;
    // }
    //if(fabs((net->output_units)[j] - (net_cpu->output_units)[j]) >= 0.01f) {
//	error=1;
  //  }
    //if(fabs((net->output_delta)[j] - (net_cpu->output_delta)[j]) >= 0.01f) {
//	error=1;
  //  }
  }

  if (error==0) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  bpnn_free(net);
  bpnn_free(net_cpu);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{

  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }


  seed = 7;
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
