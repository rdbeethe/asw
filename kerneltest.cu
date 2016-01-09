#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct timespec check_timer(const char* str, struct timespec* ts){
	struct timespec oldtime;
	// copy old time over
	oldtime.tv_nsec = ts->tv_nsec;
	oldtime.tv_sec = ts->tv_sec;
	// update ts
	clock_gettime(CLOCK_REALTIME, ts);
	// print old time
	int diffsec;
	int diffnsec;
	if(str != NULL){
		diffsec =  ts->tv_sec - oldtime.tv_sec;
		diffnsec =  ts->tv_nsec - oldtime.tv_nsec;
		// correct the values if we measured over an integer second break:
		if(diffnsec < 0){
			diffsec--;
			diffnsec += 1000000000;
		}
		printf("%s:%ds %dns\n",str,diffsec,diffnsec);
	}
	return (struct timespec) {diffsec, diffnsec};
}
 

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
	// declare timer
	struct timespec timer;

    int N = 1000000000;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);


	check_timer(NULL,&timer);
    // Initialize input vectors
    for(int i = 0; i < N; i++){
    	h_A[i] = i;
    	h_B[i] = N - i;
    }
	check_timer("Time to initialize",&timer);

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
	check_timer(NULL,&timer);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	check_timer("Time to copy to device",&timer);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
	check_timer(NULL,&timer);
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	check_timer("Time to execute kernel",&timer);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
	check_timer(NULL,&timer);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	check_timer("Time to copy back to host",&timer);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    int errors = 0;
    for(int i = 0; i < N; i++){
    	if(h_C[i] != N){
    		errors ++;
    	}

    }
    printf("checking done, errors = %d\n");

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}