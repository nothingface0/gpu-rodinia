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
/**					No longer performs "validity" function
 * to analyze	**/
/**					compactness and separation crietria;
 * instead
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
      "0: 125.35 164.62 136.07 122.05 87.94 105.61 159.17 122.31 129.44 134.11 135.93 150.37 122.81 150.26 92.34 121.67 103.49 135.81 163.08 136.06 103.46 149.15 121.38 117.21 124.72 94.99 134.16 150.57 146.14 120.23 124.95 120.48 173.79 131.66\
\n\n\
1: 155.57 85.87 101.87 122.50 136.71 133.19 131.97 113.35 116.51 169.96 141.60 86.88 143.69 159.00 146.60 112.06 113.12 126.35 130.88 142.92 136.22 141.04 123.50 157.16 139.75 107.62 104.27 147.29 111.80 88.23 135.78 170.01 115.48 118.17\
\n\n\
2: 112.32 138.15 95.19 91.61 113.21 142.87 120.67 141.46 135.55 111.18 102.90 150.93 141.35 117.18 173.99 124.52 120.98 119.04 101.77 144.91 98.31 133.08 144.66 128.56 125.39 150.76 151.56 128.72 153.41 153.74 148.04 109.33 106.89 145.60\
\n\n\
3: 91.50 88.50 147.96 162.88 137.47 109.04 127.19 161.48 140.12 148.22 112.20 130.41 124.33 94.40 110.39 109.26 137.66 133.84 114.19 118.27 127.84 104.14 118.91 111.12 131.46 115.75 117.68 109.59 97.56 106.61 110.71 142.85 123.53 154.12\
\n\n\
4: 149.36 136.19 132.80 149.05 161.83 154.11 109.11 99.18 125.11 97.73 156.67 128.40 103.66 127.04 134.17 152.35 158.30 122.04 135.02 108.73 148.45 122.78 131.25 134.10 114.72 139.48 139.76 134.05 133.27 132.13 120.03 110.11 109.80 100.37\
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
