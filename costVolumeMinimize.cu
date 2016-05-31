#include "costVolumeMinimize.h"

using namespace std;
using namespace cv;
using namespace cuda;

#define ILP_LEVEL 4
// Device code
__global__ void costVolumeMinimize_kernel(PtrStepf vol, PtrStep<unsigned char> output, int nrows, int ndisp, int ncols){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;
	gy *= ILP_LEVEL;

	// only threads which land in the image participate
	if(gy < nrows && gx < ncols){

		// this will store the disp val of the lowest cost
		int mindisp[ILP_LEVEL];
		float mincost[ILP_LEVEL];
		float* rowin[ILP_LEVEL];
		unsigned char* rowout[ILP_LEVEL];

#pragma unroll
		for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
			// arbitrary large number
			mincost[ilp] = 1e6;
		}


		// now go through each disparity
		for(int disp = 0; disp < ndisp; disp ++){
			float cost[ILP_LEVEL];
#pragma unroll
			for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
				if(gy + ilp < nrows){
					rowin[ilp] = (float*) ( (char*)vol.data + vol.step*nrows*disp + vol.step*(gy+ilp));
					rowout[ilp] = (unsigned char*) ( (char*)output.data + output.step*(gy+ilp));
				}
				__syncthreads();
			}
#pragma unroll
			for(int ilp = 0; ilp < ILP_LEVEL; ilp++){
				if(gy + ilp < nrows){
					cost[ilp] = rowin[ilp][gx];
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
			if(gy + ilp < nrows){
				rowout[ilp][gx] = mindisp[ilp];
			}
		}
	}
}

void costVolumeMinimize_gpu(GpuMat cost_volume, Mat& outim, int ndisp){
	int nrows = cost_volume.rows/ndisp;
	int ncols = cost_volume.cols;
	// allocate output matrix on gpu
	GpuMat d_output(nrows,ncols,CV_8UC1);

	// settings for the kernel
	// trying to use 128 threads-wide so the uchar global write is 128 bytes
	dim3 threadsPerBlock(128,1);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y/ILP_LEVEL+1);
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    costVolumeMinimize_kernel<<<blocksPerGrid, threadsPerBlock>>>(cost_volume, d_output, nrows, ncols, ndisp);
	cudaDeviceSynchronize();
    check_timer("costVolumeMinimize_gpu time",&timer);
	gpu_perror("costVolumeMinimize_kernel");

	// copy debug back over
	d_output.download(outim);

	// cleanup the temporary image memory
	d_output.release();
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
			for(int disp = 1; disp < ndisp; disp++){
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

