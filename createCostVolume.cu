#include "cost_volume.h"
#include "createCostVolume.h"
#include "timer.h"
#include "helper.h"

using namespace std;
using namespace cv;

// Device code
__global__ void createCostVolume_kernel(int* ref_global, int* tgt_global, struct cost_volume_t vol, int* debug){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;

	extern __shared__ int tgt_data[]; // contains relevant tgt image data

	// copy target image global memory into shared memory (all threads must participate)
	for(int i = 0; i < vol.ndisp + blockDim.x; i += blockDim.x){
		// check to make sure the actual read lands in 0 <= col < ncols  && row < nrows
		if(gy < vol.nrows && (gx - (vol.ndisp-1) + i) >= 0 && (gx - (vol.ndisp-1) + i) < vol.ncols){
			tgt_data[(blockDim.x + vol.ndisp - 1)*threadIdx.y + threadIdx.x + i] = tgt_global[vol.ncols*gy + gx - (vol.ndisp-1) + i];
		}
		__syncthreads();
	}

	// now only threads which land in the image participate
	if(gy < vol.nrows && gx < vol.ncols){

		// get reference pixel from global memory
		int ref = ref_global[vol.ncols*gy + gx];

		// pull out channel data from reference pixel (brought in as an int)
		int rr,rg,rb;
		rr = (ref&0x000000FF) >> 0;
		rb = (ref&0x0000FF00) >> 8;
		rg = (ref&0x00FF0000) >> 16;
		
		// now go through each disparity
		for(int disp = 0; disp < vol.ndisp; disp ++){
			float cost;
			// check if this disp has a pixel in the tgt image
			if( gx - disp >= 0){
				// read tgt pixel from shared memory
				int tgt = tgt_data[(blockDim.x + vol.ndisp - 1)*threadIdx.y + (vol.ndisp-1) + threadIdx.x - disp];

				// separate channel data
				int tr,tg,tb;
				tr = (tgt&0x000000FF) >> 0;
				tb = (tgt&0x0000FF00) >> 8;
				tg = (tgt&0x00FF0000) >> 16;

				// using SAD for aggregate cost function
				cost = abs(rr - tr) + abs(rb-tb) + abs(rg-tg);
			}else{
				// these values of the cost volume don't correspond to two real pixels, so make the cost high
				cost = 9999;
			}
			__syncthreads();
			// now write the cost to the actual cost_volume
			vol.volume[vol.stride*vol.nrows*disp + vol.stride*gy + gx] = cost;
		}
	}
}

struct cost_volume_t createCostVolume_gpu(Mat leftim, Mat rightim, int ndisp){
	int nchans = leftim.channels();
	int nrows = leftim.rows;
	int ncols = leftim.cols;
	// find stride so that rows in global memory align to 128-byte boundaries
	int boundary = 128/sizeof(float);
	int stride = ncols + (boundary - ncols%boundary)%boundary;
	// allocate gpu memory for cost volume
	float* volume_gpu;
	cudaMalloc(&volume_gpu,nrows*ndisp*stride*sizeof(float));
	// zero the volume_gpu
	// gpu_memset<<<ncols*ndisp*stride*sizeof(float)/1024 + 1, 1024>>>((unsigned char*)volume_gpu,0,ncols*ndisp*stride*sizeof(float));
	// gpu_perror("memset on volume");
	// init struct cost_volume_t object
	struct cost_volume_t cost_volume = {volume_gpu,nrows,ncols,ndisp,stride};
	// convert BGR images to RGBA
	cvtColor(leftim,leftim,CV_BGR2RGBA);
	cvtColor(rightim,rightim,CV_BGR2RGBA);
	// copy left image to to GPU
	unsigned char* d_im_l;
	cudaMalloc(&d_im_l, 4*nrows*ncols*sizeof(unsigned char));
    cudaMemcpy(d_im_l, leftim.data, 4*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
	// copy right image to to GPU
	unsigned char* d_im_r;
	cudaMalloc(&d_im_r, 4*nrows*ncols*sizeof(unsigned char));
    cudaMemcpy(d_im_r, rightim.data, 4*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
	// debug setup
	Mat debug(nrows,ncols,CV_8UC4);
	unsigned char* d_debug;
	cudaMalloc(&d_debug,nrows*ncols*sizeof(int));
	// zero the volume_gpu
	// gpu_memset<<<ncols*nrows*sizeof(int)/1024 + 1, 1024>>>((unsigned char*)d_debug,0,ncols*nrows*sizeof(int));
	// gpu_perror("memset on debug");

	// settings for the kernel
	// should be 32-threads wide to ensure 128-byte block global reads
	dim3 threadsPerBlock(32,4);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y+1);
	int tgt_shared_mem = (threadsPerBlock.x+ndisp-1)*threadsPerBlock.y*sizeof(int);
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    createCostVolume_kernel<<<blocksPerGrid, threadsPerBlock, tgt_shared_mem>>>((int*)d_im_l, (int*)d_im_r, cost_volume, (int*)d_debug);
	cudaDeviceSynchronize();
    check_timer("cost_volume_gpu time",&timer);
	gpu_perror("createCostVolume_kernel");

	// copy debug back over
    cudaMemcpy((int*)debug.data, d_debug, nrows*ncols*sizeof(int), cudaMemcpyDeviceToHost);
	// imshow("window",leftim); waitKey(0);
	// imshow("window",debug); waitKey(0);
	// imshow("window",leftim); waitKey(0);

	// cleanup the temporary image memory
	cudaFree(d_im_l);
	cudaFree(d_im_r);
	cudaFree(d_debug);

	return cost_volume;
}

struct cost_volume_t createCostVolume(Mat leftim, Mat rightim,int ndisp){
	int nchans = leftim.channels();
	int nrows = leftim.rows;
	int ncols = leftim.cols;
	int stride = ncols;
	float* volume = (float*)malloc(ncols*nrows*nchans*ndisp*sizeof(float));
	// init struct cost_volume_t object
	struct cost_volume_t cost_volume = {volume,nrows,ncols,ndisp,stride};

	// make sure images are the same size
	if(leftim.cols != rightim.cols || leftim.rows != rightim.rows && leftim.channels() == rightim.channels()){
		printf("ERROR: left and right images in createCostVolume do not have matching rows and cols and channels\n");
		return cost_volume;
	}

	struct timespec timer;
	check_timer(NULL,&timer);

	unsigned char* left =  (unsigned char*)leftim.data;
	unsigned char* right = (unsigned char*)rightim.data;
	// init values to very large numbers
	// the reason for this is that some regions near volume edges won't be dealt with
	for( int i = 0; i < ncols*nrows*nchans*ndisp; i++){
		// arbitrary large number
		volume[i] = 9999;
	}

	// organization will be ndisp images of rows of pixels
	// iterate over the whole image
	for(int col = 0; col < ncols; col++){
		for(int row = 0; row < nrows; row++){
			// iterate over the disparities
			for(int disp = 0; disp < min(ndisp,col+1); disp++){
				// get difference over channels
				float diff = 0;
				for(int chan = 0; chan < nchans; chan++){
					diff += abs(left[(ncols*row + col)*nchans + chan] - right[(ncols*row + col - disp)*nchans + chan]);
				}
				volume[nrows*ncols*disp + ncols*row + col] = diff;
			}
		}
	}
	check_timer("createCostVolume",&timer);
	return cost_volume;
}

