#include "cost_volume.h"
#include "createCostVolume.h"
#include "timer.h"
#include "helper.h"

using namespace std;
using namespace cv;
using namespace cuda;

// Device code
__global__ void createCostVolume_kernel(PtrStepi ref_global, PtrStepi tgt_global, PtrStepf volume, int nrows, int ncols, int ndisp){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;

	extern __shared__ struct rgba_pixel tgt_data[]; // contains relevant tgt image data

	// copy target image global memory into shared memory (all threads must participate)
	for(int i = 0; i < ndisp + blockDim.x; i += blockDim.x){
		struct rgba_pixel* row = (struct rgba_pixel*)(((char*)tgt_global.data) + tgt_global.step*gy);
		// check to make sure the actual read lands in 0 <= col < ncols  && row < nrows
		if(gy < nrows && (gx - (ndisp-1) + i) >= 0 && (gx - (ndisp-1) + i) < ncols && threadIdx.x + i < ndisp + blockDim.x - 1){
			tgt_data[(blockDim.x + ndisp - 1)*threadIdx.y + threadIdx.x + i] = row[gx - (ndisp-1) + i];
		}
		__syncthreads();
	}

	// now only threads which land in the image participate
	if(gy < nrows && gx < ncols){
		// get reference pixel from global memory
		struct rgba_pixel* row = (struct rgba_pixel*)(((char*)ref_global.data) + ref_global.step*gy);
		struct rgba_pixel ref = row[gx];

		// debug
		// int debug_x = 160;
		// int debug_y = 114;
		// if(gx == debug_x && gy == debug_y){
		// 	printf("blockId = %d,%d\n",blockIdx.x,blockIdx.y);
		// 	printf("ref = %d,%d,%d\n",ref.b,ref.g,ref.r);
		// }

		// now go through each disparity
		for(int disp = 0; disp < ndisp; disp ++){
			float cost;
			// check if this disp has a pixel in the tgt image
			if( gx - disp >= 0){
				// read tgt pixel from shared memory
				struct rgba_pixel tgt = tgt_data[(blockDim.x + ndisp - 1)*threadIdx.y + (ndisp-1) + threadIdx.x - disp];

				// using SAD for aggregate cost function
				cost = abs(ref.r - tgt.r) + abs(ref.b-tgt.b) + abs(ref.g-tgt.g);

				// debug
				//if(gx == debug_x && gy == debug_y){
				//	printf("ref = %d,%d,%d\t",ref.b,ref.g,ref.r);
				//	printf("tgt= %d,%d,%d\t",tgt.b,tgt.g,tgt.r);
				//	printf("disp,cost = %d,%f\t",disp,cost);
				//	printf("shared index = %d\n",(blockDim.x + ndisp - 1)*threadIdx.y + (ndisp-1) + threadIdx.x - disp);
				//}
			}else{
				// these values of the cost volume don't correspond to two real pixels, so make the cost high
				cost = 9999;
			}
			__syncthreads();
			// now write the cost to the actual cost_volume
			float* row = (float*)(((char*)volume.data) + nrows*volume.step*disp + volume.step*gy);
			row[gx] = cost;
		}
	}
}

GpuMat createCostVolume_gpu(Mat leftim, Mat rightim, int ndisp){
	if(leftim.type() != rightim.type() || leftim.rows != rightim.rows || leftim.cols != rightim.cols){
		printf("ERROR, in createCostVolume_gpu(), leftim and rightim do not match in type, rows, and cols\n");
		return GpuMat();
	}
	if(leftim.type() != CV_8UC3){
		printf("ERROR, in createCostVolume_gpu(), leftim must have type CV_8UC3 in current implementation\n");
		return GpuMat();
	}
	int nchans = leftim.channels();
	int nrows = leftim.rows;
	int ncols = leftim.cols;
	// allocate gpu memory for cost volume
	GpuMat volume(nrows*ndisp,ncols,CV_32FC1);
	// convert BGR images to RGBA
	cvtColor(leftim,leftim,CV_BGR2RGBA);
	cvtColor(rightim,rightim,CV_BGR2RGBA);
	// create GpuMat objects from leftim and rightim
	// copy left image to to GPU
	GpuMat d_im_l;
	d_im_l.upload(leftim);
	// copy right image to to GPU
	GpuMat d_im_r;
	d_im_r.upload(rightim);

	// settings for the kernel
	// should be 32-threads wide to ensure 128-byte block global reads
	dim3 threadsPerBlock(32,4);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y+1);
	int tgt_shared_mem = (threadsPerBlock.x+ndisp-1)*threadsPerBlock.y*sizeof(int);
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    createCostVolume_kernel<<<blocksPerGrid, threadsPerBlock, tgt_shared_mem>>>(d_im_l, d_im_r, volume, nrows, ncols, ndisp);
	cudaDeviceSynchronize();
    check_timer("cost_volume_gpu time",&timer);
	gpu_perror("createCostVolume_kernel");

	// cleanup the gpu memory
	d_im_l.release();
	d_im_r.release();

	return volume;
}

struct cost_volume_t cost_volume_from_gpumat(GpuMat gpumat, int ndisp){
	float* volume = (float*)gpumat.data;
	int nrows = gpumat.rows/ndisp;
	int ncols = gpumat.cols;
	int stride = gpumat.step / sizeof(float);
	struct cost_volume_t cv = {volume,nrows,ncols,ndisp,stride};
	return cv;
}

struct cost_volume_t cost_volume_from_mat(Mat mat, int ndisp){
	float* volume = (float*)mat.data;
	int nrows = mat.rows/ndisp;
	int ncols = mat.cols;
	int stride = mat.cols;
	struct cost_volume_t cv = {volume,nrows,ncols,ndisp,stride};
	return cv;
}

struct bgr_pixel {
	unsigned char b;
	unsigned char g;
	unsigned char r;
};

cv::Mat createCostVolume(Mat leftim, Mat rightim, int ndisp){
	// make sure images are the same size
	if(leftim.cols != rightim.cols || leftim.rows != rightim.rows && leftim.type() == rightim.type()){
		printf("ERROR: in createCostVolume(), left and right images do not have matching rows and cols and type\n");
		return Mat();
	}
	if(leftim.type() != CV_8UC3){
		printf("ERROR: in createCostVolume(), leftim must have type CV_8UC3 in current implementation\n");
		return Mat();
	}
	int nchans = leftim.channels();
	int nrows = leftim.rows;
	int ncols = leftim.cols;

	// initialize the cost_volume Mat
	Mat volume(nrows*ndisp,ncols,CV_32FC1);

	struct bgr_pixel* left =  (struct bgr_pixel*)leftim.data;
	struct bgr_pixel* right = (struct bgr_pixel*)rightim.data;

	struct timespec timer;
	check_timer(NULL,&timer);

	// organization will be ndisp images of rows of pixels
	// iterate over the whole image
	for(int col = 0; col < ncols; col++){
		for(int row = 0; row < nrows; row++){
			// iterate over the disparities
			struct bgr_pixel ref = left[ncols*row + col];

			// debug
			// int debug_x = 160;
			// int debug_y = 114;
			// if(col == debug_x && row == debug_y){
			// 	printf("ref = %d,%d,%d\n",ref.b,ref.g,ref.r);
			// }

			for(int disp = 0; disp < ndisp; disp++){
				if(col - disp >= 0){
					// get difference over channels
					float diff = 0;
					struct bgr_pixel tgt = right[ncols*row + col-disp];
					diff = abs((int)ref.r - (int)tgt.r) + abs((int)ref.g - (int)tgt.g) + abs((int)ref.b - (int)tgt.b);

					// debug
					// if(col == debug_x && row == debug_y){
					// 	printf("tgt = %d,%d,%d\t",tgt.b,tgt.g,tgt.r);
					// 	printf("disp,cost = %d,%f\n",disp,diff);
					// }

					((float*)volume.data)[nrows*ncols*disp + ncols*row + col] = diff;
				}else{
					// no pair of valid pixels at this disp, assign an arbitrary large number
					((float*)volume.data)[nrows*ncols*disp + ncols*row + col] = 9999;
				}
			}
		}
	}
	check_timer("createCostVolume",&timer);
	return volume;
}

