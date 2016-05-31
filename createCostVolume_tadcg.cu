#include "cost_volume.h"
#include "createCostVolume_tadcg.h"
#include "timer.h"
#include "helper.h"

using namespace std;
using namespace cv;

struct two16s {
	short int a;
	short int b;
};

// Device code
__global__ void createCostVolume_tadcg_kernel(cuda::PtrStepi ref_global, cuda::PtrStepi tgt_global, struct cost_volume_t vol, cuda::PtrStepi debug, float tc, float tg, float alpha){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;

	extern __shared__ struct rgba_pixel tgt_data[]; // contains relevant tgt image data

	// set shared row pointer
	struct rgba_pixel* s_row = (struct rgba_pixel*)((char*)tgt_data + (blockDim.x+vol.ndisp)*threadIdx.y*sizeof(struct rgba_pixel));
	
	{
		// set global row for data transfer loop
		struct rgba_pixel* g_row = (struct rgba_pixel*)((char*)tgt_global.data + tgt_global.step*gy);

		// copy target image global memory into shared memory (all threads must participate)
		for(int i = 0; i < vol.ndisp + blockDim.x; i += blockDim.x){
			// check to make sure the actual read lands in 0 <= col < ncols  && row < nrows
			if(gy < vol.nrows && (gx - vol.ndisp + i) >= 0 && (gx - vol.ndisp + i) < vol.ncols && threadIdx.x + i < vol.ndisp + blockDim.x){
				s_row[threadIdx.x + i] = g_row[gx - vol.ndisp + i];
			}
			__syncthreads();
		}
	}

	// now only threads which land in the image participate
	if(gy < vol.nrows && gx < vol.ncols){
		struct rgba_pixel ref0, ref;

		{
			struct rgba_pixel* g_row = (struct rgba_pixel*)((char*)ref_global.data + ref_global.step*gy);

			// get reference pixels from global memory
			ref = g_row[gx];
			// ref0 is the previous pixel, for the gradient calculation
			// casting rgba_pixel to int allows multiplying by (gx>0) which avoids a divergence opportunity
			((int*)&ref0)[0] = (((int*)g_row)[max(gx-1,0)]) * (gx > 0);
			// old, divergent code (easier to understand:
			// if(gx > 0){
			// 	ref0 = g_row[gx-1];
			// }else{
			// 	((int*)&(ref0))[0] = 0;
			// }
		} 

		struct rgba_pixel tgt;
		struct rgba_pixel tgt0;
		// if(gx == 100 && gy == 100){
		// 	printf("ref,ref0 = %d,%d,%d %d,%d,%d\n",ref.b,ref.g,ref.r,ref0.b,ref0.g,ref0.r);
		// }

		// now go through each disparity
		for(int disp = 0; disp < vol.ndisp; disp ++){
			float* g_row = (float*)((char*)vol.volume + (disp*vol.nrows+gy)*vol.stride*sizeof(float));
			float cost;
			int adc, adg;
			// check if this disp has a pixel in the tgt image
			if( gx - disp >= 0){
				// read tgt pixel from shared memory
				tgt = s_row[vol.ndisp + threadIdx.x - disp];
				// tgt0 is for calculating the gradient
				tgt0 = s_row[vol.ndisp-1 + threadIdx.x - disp];

				// this is the CUDA-C way to do this
				// caluculate absolute difference of color
				adc = abs(ref.r - tgt.r) + abs(ref.g - tgt.g) + abs(ref.b - tgt.b);
				// caluculate absolute difference of gradient
				adg = abs(ref.r-ref0.r - tgt.r+tgt0.r) + abs(ref.g-ref0.g - tgt.g+tgt0.g) + abs(ref.b-ref0.b - tgt.b+tgt0.b);

				// this is the PTX way to do this
				// although SIMD assembly instructions show a slight performance improvement, though these instructions are Kepler-specific
				// int C = 0;
				// int rgrad;
				// int tgrad;
				// //calculate gradients
				// asm("vsub4.s32.u32.u32.sat" " %0, %1, %2, %3;": "=r" (rgrad):"r" (((int*)&ref)[0]), "r" (((int*)&ref0)[0]), "r" (C));
				// asm("vsub4.s32.u32.u32.sat" " %0, %1, %2, %3;": "=r" (tgrad):"r" (((int*)&tgt)[0]), "r" (((int*)&tgt0)[0]), "r" (C));
				// // caluculate absolute difference of color
				// asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r" (adc):"r" (((int*)&ref)[0]), "r" (((int*)&tgt)[0]), "r" (C));
				// // caluculate absolute difference of gradient
				// asm("vabsdiff4.u32.s32.s32.add" " %0, %1, %2, %3;": "=r" (adg):"r" (rgrad), "r" (tgrad), "r" (C));


				// calculate cost with TAD C+G
				cost = alpha*min(tc,(float)adc)+(1-alpha)*min(tg,(float)adg);
			}else{
				// these values of the cost volume don't correspond to two real pixels, so make the cost high
				cost = 9999;
			}
			__syncthreads();
			// now write the cost to the actual cost_volume
			g_row[gx] = cost;
			// if(gx == 100 && gy == 100){
			// 	printf("tgt,tgt0 = %d,%d,%d %d,%d,%d\t",tgt.b,tgt.g,tgt.r,tgt0.b,tgt0.g,tgt0.r);
			// 	printf("disp,cost,adc,adg = %d,%f,%f,%f\n",disp,cost,adc,adg);
			// }
			__syncthreads();
		}
	}
}


struct cost_volume_t createCostVolume_tadcg_gpu(Mat leftim, Mat rightim, int ndisp, float tc, float tg, float alpha){
	int nchans = leftim.channels();
	int nrows = leftim.rows;
	int ncols = leftim.cols;
	size_t pitch;
	// allocate gpu memory for cost volume
	float* volume_gpu;
	cudaMallocPitch(&volume_gpu,&pitch,ncols*sizeof(float),ndisp*nrows);
	int stride = pitch / sizeof(float);
	// init struct cost_volume_t object
	struct cost_volume_t cost_volume = {volume_gpu,nrows,ncols,ndisp,stride};
	// convert BGR images to RGBA
	cvtColor(leftim,leftim,CV_BGR2RGBA);
	cvtColor(rightim,rightim,CV_BGR2RGBA);
	// copy left image to to GPU
	cuda::GpuMat d_im_l;
	d_im_l.upload(leftim);
	// copy right image to to GPU
	cuda::GpuMat d_im_r;
	d_im_r.upload(rightim);
	// debug setup
	cuda::GpuMat d_debug(Size(ncols,nrows),CV_8UC4);

	// settings for the kernel
	// should be 32-threads wide to ensure 128-byte block global reads
	dim3 threadsPerBlock(32,4);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y+1);
	int tgt_shared_mem = (threadsPerBlock.x+ndisp)*threadsPerBlock.y*sizeof(int);
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    createCostVolume_tadcg_kernel<<<blocksPerGrid, threadsPerBlock, tgt_shared_mem>>>(d_im_l, d_im_r, cost_volume, d_debug, tc,tg,alpha);
	cudaDeviceSynchronize();
    check_timer("createCostVolume_tadcg_gpu time",&timer);
	gpu_perror("createCostVolume_tadcg_kernel");

	// copy debug back over
	Mat debug;
	//d_debug.download(debug);
	// imshow("window",leftim); waitKey(0);
	// imshow("window",debug); waitKey(0);
	// imshow("window",leftim); waitKey(0);

	// cleanup the temporary image memory
	d_im_l.release();
	d_im_r.release();
	d_debug.release();

	return cost_volume;
}


struct bgr_pixel {
	unsigned char b;
	unsigned char g;
	unsigned char r;
};

struct cost_volume_t createCostVolume_tadcg(Mat leftim, Mat rightim, int ndisp, float tc, float tg, float alpha){
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
			struct bgr_pixel ref,ref0, tgt,tgt0;
			float cost;
			ref = ((struct bgr_pixel*)(left))[ncols*row + col];
			if(col >0){
				ref0 = ((struct bgr_pixel*)(left))[ncols*row + col - 1];
			}else{
				ref0.b = 0;
				ref0.g = 0;
				ref0.r = 0;
			}
			// if(col == 100 && row == 100){
			// 	printf("ref,ref0 = %d,%d,%d %d,%d,%d\n",ref.b,ref.g,ref.r,ref0.b,ref0.g,ref0.r);
			// }
			// iterate over the disparities
			for(int disp = 0; disp < min(ndisp,col+1); disp++){
				// get absolute difference of color and of grad
				float adc = 0;
				float adg = 0;
				tgt = ((struct bgr_pixel*)(right))[ncols*row + col-disp];
				if(col > 0){
					tgt0 = ((struct bgr_pixel*)(right))[ncols*row + col-disp - 1];
				}else{
					tgt0.b = 0;
					tgt0.g = 0;
					tgt0.r = 0;
				}

				// caluculate absolute difference of color
				adc = abs((int)ref.r - (int)tgt.r) + abs((int)ref.g - (int)tgt.g) + abs((int)ref.b - (int)tgt.b);

				// caluculate absolute difference of gradient
				adg = abs((int)ref.r-(int)ref0.r - (int)tgt.r+(int)tgt0.r) + abs((int)ref.g-(int)ref0.g - (int)tgt.g+(int)tgt0.g) + abs((int)ref.b-(int)ref0.b - (int)tgt.b+(int)tgt0.b);

				// calculate cost with TAD C+G
				cost = alpha*min(adc,tc) + (1-alpha)*min(adg,tg);

				// if(col == 100 && row == 100){
				// 	printf("tgt,tgt0 = %d,%d,%d %d,%d,%d\t",tgt.b,tgt.g,tgt.r,tgt0.b,tgt0.g,tgt0.r);
				// 	printf("disp,cost,adc,adg = %d,%f,%f,%f\n",disp,cost,adc,adg);
				// }
				volume[nrows*ncols*disp + ncols*row + col] = cost;
			}
		}
	}
	check_timer("createCostVolume_tadcg time",&timer);
	return cost_volume;
}
