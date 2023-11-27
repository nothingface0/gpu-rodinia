/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**
 * **/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					No longer performs "validity" function to
 * analyze	**/
/**					compactness and separation crietria; instead
 * **/
/**					calculate root mean squared error.
 * **/
/**                                                                     **/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 1

#include "kmeans.h"
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
  char *help = "\nUsage: %s [switches] -i filename\n\n"
               "    -i filename      :file containing data to be clustered\n"
               "    -m max_nclusters :maximum number of clusters allowed    "
               "[default=5]\n"
               "    -n min_nclusters :minimum number of clusters allowed    "
               "[default=5]\n"
               "    -t threshold     :threshold value                       "
               "[default=0.001]\n"
               "    -l nloops        :iteration for each number of clusters "
               "[default=1]\n"
               "    -b               :input file is in binary format\n"
               "    -r               :calculate RMSE                        "
               "[default=off]\n"
               "    -o               :output cluster center coordinates     "
               "[default=off]\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
  int opt;
  extern char *optarg;
  char *filename = 0;
  float *buf;
  char line[1024];
  int isBinaryFile = 0;

  float threshold = 0.001; /* default value */
  int max_nclusters = 5;   /* default value */
  int min_nclusters = 5;   /* default value */
  int best_nclusters = 0;
  int nfeatures = 0;
  int npoints = 0;
  float len;

  float **features;
  float **cluster_centres = NULL;
  int i, j, index;
  int nloops = 1; /* default value */

  int isRMSE = 0;
  float rmse;

  int isOutput = 0;
  // float	cluster_timing, io_timing;

  char *results =
      "0: 140.45 134.60 120.53 101.62 144.25 118.62 102.42 94.80 113.94 138.95 143.07 145.97 149.62 124.89 151.08 132.04 124.40 173.02 130.36 141.38 111.53 137.85 134.12 150.70 93.41 138.47 116.49 122.45 140.81 103.86 127.38 148.80 143.87 116.78\
\n\n\
1: 88.38 89.17 117.35 168.93 95.18 123.13 157.62 128.60 139.39 154.31 131.44 115.80 152.69 121.15 128.02 159.34 129.66 121.80 144.83 139.08 150.30 125.04 151.74 99.82 123.81 118.54 149.97 146.47 111.13 102.10 137.84 82.87 134.99 134.47\
\n\n\
2: 135.49 131.12 145.03 156.30 115.21 131.91 107.39 154.18 115.16 114.34 129.65 155.55 130.64 116.66 132.72 97.76 158.18 74.92 143.29 131.84 138.67 107.02 93.58 163.04 124.72 156.50 144.56 142.00 121.47 143.79 133.52 131.60 98.28 142.31\
\n\n\
3: 106.16 154.39 128.49 118.84 124.07 146.73 155.51 171.68 147.06 118.55 97.57 126.61 100.50 148.08 99.02 150.76 116.63 137.09 118.96 114.80 142.81 145.32 117.87 148.31 145.87 126.89 101.03 89.42 151.22 123.58 100.94 120.66 121.97 132.94\
\n\n\
4: 156.86 130.98 121.34 119.50 159.97 105.90 138.13 107.66 119.71 112.10 136.99 87.61 111.50 138.76 117.56 128.19 113.54 101.73 109.10 97.57 109.61 120.52 139.18 97.28 158.07 124.35 128.78 144.82 108.12 162.73 115.67 132.85 123.28 92.64\
\n\n";

  /* obtain command line arguments and change appropriate options */
  while ((opt = getopt(argc, argv, "i:t:m:n:l:bro")) != EOF) {
    switch (opt) {
    case 'i':
      filename = optarg;
      break;
    case 'b':
      isBinaryFile = 1;
      break;
    case 't':
      threshold = atof(optarg);
      break;
    case 'm':
      max_nclusters = atoi(optarg);
      break;
    case 'n':
      min_nclusters = atoi(optarg);
      break;
    case 'r':
      isRMSE = 1;
      break;
    case 'o':
      isOutput = 1;
      break;
    case 'l':
      nloops = atoi(optarg);
      break;
    case '?':
      usage(argv[0]);
      break;
    default:
      usage(argv[0]);
      break;
    }
  }

  if (filename == 0)
    usage(argv[0]);

  /* ============== I/O begin ==============*/
  /* get nfeatures and npoints */
  // io_timing = omp_get_wtime();
  if (isBinaryFile) { // Binary file input
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    read(infile, &npoints, sizeof(int));
    read(infile, &nfeatures, sizeof(int));

    /* allocate space for features[][] and read attributes of all objects */
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    features = (float **)malloc(npoints * sizeof(float *));
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;

    read(infile, buf, npoints * nfeatures * sizeof(float));

    close(infile);
  } else {
    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
      if (strtok(line, " \t\n") != 0)
        npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first attribute): nfeatures = 1; */
        while (strtok(NULL, " ,\t\n") != NULL)
          nfeatures++;
        break;
      }
    }

    /* allocate space for features[] and read attributes of all objects */
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    features = (float **)malloc(npoints * sizeof(float *));
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL)
        continue;
      for (j = 0; j < nfeatures; j++) {
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(infile);
  }
  // io_timing = omp_get_wtime() - io_timing;

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", npoints);
  printf("Number of features: %d\n", nfeatures);
  /* ============== I/O end ==============*/

  // error check for clusters
  if (npoints < min_nclusters) {
    printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
           min_nclusters, npoints);
    exit(0);
  }

  srand(7); /* seed for future random number generator */
  memcpy(
      features[0], buf,
      npoints * nfeatures *
          sizeof(
              float)); /* now features holds 2-dimensional array of features */
  free(buf);

  /* ======================= core of the clustering ===================*/

  // cluster_timing = omp_get_wtime();		/* Total clustering time */
  cluster_centres = NULL;
  index = cluster(npoints,       /* number of data points */
                  nfeatures,     /* number of features for each point */
                  features,      /* array: [npoints][nfeatures] */
                  min_nclusters, /* range of min to max number of clusters */
                  max_nclusters, threshold, /* loop termination factor */
                  &best_nclusters,  /* return: number between min and max */
                  &cluster_centres, /* return: [best_nclusters][nfeatures] */
                  &rmse,            /* Root Mean Squared Error */
                  isRMSE,           /* calculate RMSE */
                  nloops); /* number of iteration for each number of clusters */

  // cluster_timing = omp_get_wtime() - cluster_timing;

  /* =============== Command Line Output =============== */

  /* cluster center coordinates
     :displayed only for when k=1*/
  char *str_result = malloc(sizeof(char) * (strlen(results) + 1));
  str_result[0] = 0;
  if ((min_nclusters == max_nclusters) && (isOutput == 0)) {
    printf("\n================= Centroid Coordinates =================\n");
    for (i = 0; i < max_nclusters; i++) {
      printf("%d:", i);
      sprintf(str_result + strlen(str_result), "%d:", i);
      for (j = 0; j < nfeatures; j++) {
        printf(" %.2f", cluster_centres[i][j]);
        sprintf(str_result + strlen(str_result), " %.2f",
                cluster_centres[i][j]);
      }
      printf("\n\n");
      sprintf(str_result + strlen(str_result), "\n\n");
    }
  }

  if (strcmp(results, str_result) == 0) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED\n");
  }

  len = (float)((max_nclusters - min_nclusters + 1) * nloops);

  printf("Number of Iteration: %d\n", nloops);
  // printf("Time for I/O: %.5fsec\n", io_timing);
  // printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

  if (min_nclusters != max_nclusters) {
    if (nloops != 1) { // range of k, multiple iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    } else { // range of k, single iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    }
  } else {
    if (nloops != 1) { // single k, multiple iteration
      // printf("Average Clustering Time: %.5fsec\n",
      //		cluster_timing / nloops);
      if (isRMSE) // if calculated RMSE
        printf("Number of trials to approach the best RMSE of %.3f is %d\n",
               rmse, index + 1);
    } else {      // single k, single iteration
      if (isRMSE) // if calculated RMSE
        printf("Root Mean Squared Error: %.3f\n", rmse);
    }
  }

  /* free up memory */
  free(features[0]);
  free(features);
  return (0);
}
