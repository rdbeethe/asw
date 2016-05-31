#include "helper.h"
#include <stdio.h>

// little bitty kernel to initialize blocks of device memory
__global__ void gpu_memset(unsigned char* start, unsigned char value, int length){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int gx = bx*blockDim.x + tx;
	if(gx < length){
		start[gx] = value;
	}
}

// teeny little helper function
void gpu_perror(const char* input){
	printf("%s: %s\n", input, cudaGetErrorString(cudaGetLastError()));
}

