/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid + blockIdx.y*32768])
	{
		g_graph_mask[tid+ blockIdx.y*32768]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
				{
				g_cost[id + blockIdx.y*32768]=g_cost[tid + blockIdx.y*32768]+1;
				g_updating_graph_mask[id + blockIdx.y*32768]=true;
				}
			}
	}
}

/* START of Lishan add */

__global__ void prepare_dup_int(int* a, int size)
{
	
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;
    a[tid+size] = a[tid];
    a[tid+size*2] = a[tid];
}
 
__global__ void check_correctness_int(int* result, int size)
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
//          printf ("correcting tid=%d %d %d %d\n", tid,result[tid], result[tid+size], result[tid+size*2]);  
            result[tid] = result[tid+size*2];
        }
    }
}



__global__ void prepare_dup_bool(bool* a, int size)
{
	
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;
    a[tid+size] = a[tid];
    a[tid+size*2] = a[tid];
}
 
__global__ void check_correctness(bool* result, int size)
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
//          printf ("bool correcting tid=%d %d %d %d\n", tid,result[tid], result[tid+size], result[tid+size*2]);  
            result[tid] = result[tid+size*2];
        }
    }
}

/* END of Lishan add */



#endif 
