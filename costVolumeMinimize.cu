#include "costVolumeMinimize.h"

using namespace std;
using namespace cv;

#define ILP_LEVEL 4
// Device code
__global__ void costVolumeMinimize_kernel(struct cost_volume_t vol, unsigned char* output){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;
	gy *= ILP_LEVEL;

	// only threads which land in the image participate
	if(gy < vol.nrows && gx < vol.ncols){

		// this will store the disp val of the lowest cost
		int mindisp[ILP_LEVEL];
		float mincost[ILP_LEVEL];
#pragma unroll
		for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
			// arbitrary large number
			mincost[ilp] = 1e6;
		}


		// now go through each disparity
		for(int disp = 0; disp < vol.ndisp; disp ++){
			float cost[ILP_LEVEL];
#pragma unroll
			for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
				if(gy + ilp < vol.nrows){
					cost[ilp] = vol.volume[vol.stride*vol.nrows*disp + vol.stride*(gy+ilp) + gx];
				}
				__syncthreads();
			}
#pragma unroll
			for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
				if(cost[ilp] < mincost[ilp]){
					mincost[ilp] = cost[ilp];
					mindisp[ilp] = disp;
				}
			}
			__syncthreads();
		}

		// write the resulting minimum to the output
#pragma unroll
		for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
			if(gy + ilp < vol.nrows){
				output[vol.ncols*(gy+ilp) + gx] = mindisp[ilp];
			}
		}
	}
}

void costVolumeMinimize_gpu(struct cost_volume_t cost_volume, Mat& outim){
	int nrows = cost_volume.nrows;
	int ncols = cost_volume.ncols;
	// init out mat
	outim = Mat::zeros(nrows,ncols,CV_8U);
	// allocate output matrix on gpu
	unsigned char* d_output;
	cudaMalloc(&d_output, nrows*ncols*sizeof(unsigned char));
	// zero the d_output
	// gpu_memset<<<nrows*ncols*sizeof(unsigned char)/1024 + 1, 1024>>>((unsigned char*)d_output,0,nrows*ncols*sizeof(unsigned char));
	// gpu_perror("memset on output");

	// settings for the kernel
	// trying to use 128 threads-wide so the uchar global write is 128 bytes
	dim3 threadsPerBlock(128,1);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y/ILP_LEVEL+1);
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    costVolumeMinimize_kernel<<<blocksPerGrid, threadsPerBlock>>>(cost_volume, (unsigned char*)d_output);
	cudaDeviceSynchronize();
    check_timer("costVolumeMinimize_gpu time",&timer);
	gpu_perror("costVolumeMinimize_kernel");

	// copy debug back over
    cudaMemcpy((unsigned char*)outim.data, d_output, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	// imshow("window",outim); waitKey(0);

	// cleanup the temporary image memory
	cudaFree(d_output);
}

void costVolumeMinimize(struct cost_volume_t cost_volume, Mat& outim){
	int ndisp = cost_volume.ndisp;
	int nrows = cost_volume.nrows;
	int ncols = cost_volume.ncols;
	// init out mat
	outim = Mat::zeros(nrows,ncols,CV_8U);
	unsigned char* out = (unsigned char*) (outim.data);
	float* volume = cost_volume.volume;
	for(int col = 0; col < ncols; col++){
		for(int row = 0; row < nrows; row++){
			float minval = volume[nrows*ncols*0 + ncols*row + col];
			int minidx = 0;
			// iterate over the disparities
			for(int disp = 1; disp < min(ndisp,col); disp++){
				float test = volume[nrows*ncols*disp + ncols*row + col];
				if(test < minval){
					minval = test;
					minidx = disp;
				}
			}
			out[ncols*row + col] = (unsigned char)minidx;
		}
	}
}

