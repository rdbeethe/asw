#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "timer.h"
#include "createCostVolume.h"
#include "createCostVolume_tadcg.h"
#include "costVolumeMinimize.h"
#include "costVolumeFilter_jointBilateral.h"
#include "costVolumeFilter_guided.h"
#include "costVolumeFilter_box.h"
#include "cost_volume.h"

using namespace std;
using namespace cv;
using namespace cuda;

struct cost_volume_t get_gpu_volume(struct cost_volume_t vin){
	struct cost_volume_t vout;
	vout.nrows  = vin.nrows;
	vout.ncols  = vin.ncols;
	vout.ndisp  = vin.ndisp;
	vout.stride = vin.ncols;
	// copy the gpu data directly
	float* gpu_copy = (float*)malloc(vin.stride*vin.nrows*vin.ndisp*sizeof(float));
    cudaMemcpy(gpu_copy, vin.volume, vin.stride*vin.nrows*vin.ndisp*sizeof(float), cudaMemcpyDeviceToHost);
	// now copy without padding
	vout.volume = (float*)malloc(vout.ncols*vout.nrows*vout.ndisp*sizeof(float));
	for(int col = 0; col < vout.ncols; col++){
		for(int row = 0; row < vout.nrows; row++){
			// iterate over the disparities
			for(int disp = 0; disp < vout.ndisp; disp++){
				vout.volume[vout.nrows*vout.ncols*disp + vout.ncols*row + col] = gpu_copy[vin.nrows*vin.stride*disp + vin.stride*row + col];
			}
		}
	}
	free(gpu_copy);
	return vout;
}

void viewSlices(struct cost_volume_t& cost_volume, int first, int last){
	int nrows = cost_volume.nrows;
	int stride = cost_volume.stride;
	float* vin = cost_volume.volume;
	if(last < 0){
		last = cost_volume.ndisp - last;
	}
	for(int disp = first; disp <= last; disp++){
		printf("\n%d\n",disp);
		Mat slicein(nrows,stride,CV_32F,&(vin[nrows*stride*disp]));
		double m,M; minMaxLoc(slicein,&m,&M);
		printf("min,Max of slice = %f,%f\n",m,M);
		printf("slice rows,cols: %d,%d\n",slicein.rows,slicein.cols);
		Mat temp = (slicein - m)/(M-m);
		imshow("window",temp); if((char)waitKey(0)=='q') break;
	}
}

int main(int argc, char** argv){
	cudaDeviceReset();
	// spacial and intensity sigmas
	double s_sigma, c_sigma;
	// size of bilateral kernel
	int ksize;
	// number of disparities to check
	int ndisp;
	// input images
	Mat l_im, r_im;

	if(argc < 6){
		printf("usage: %s <left image> <right image> <num disparities> <kernel size> <spacial sigma> <color sigma>\n\n",argv[0]);
		printf("... for now, using defaults (l.png r.png 64 15 5 50)\n");
		l_im = imread("l.png");
		r_im = imread("r.png");
		ndisp = 64;
		ksize = 15;
		s_sigma = 5;
		c_sigma = 50;
	}else{
		// read images, convert to floats
		l_im = imread(argv[1]);
		r_im = imread(argv[2]);
		ndisp = atoi(argv[3]);
		ksize = atoi(argv[4]);
		s_sigma = atof(argv[5]);
		c_sigma = atof(argv[6]);
	}
	printf("ndisp,ksize,s_sigma,c_sigma: %d,%d,%.3f,%.3f\n",ndisp,ksize,s_sigma,c_sigma);

	Mat out,out_gpu;
	//GpuMat gpumat_volume = createCostVolume_gpu(l_im, r_im, 64);
	GpuMat gpumat_volume = createCostVolume_tadcg_gpu(l_im, r_im, 64,20,90,.9);
	//struct cost_volume_t gpu_volume = cost_volume_from_gpumat(gpumat_volume, 64);
	//costVolumeFilter_jointBilateral_gpu(gpu_volume, l_im, ksize, c_sigma, s_sigma);
	//costVolumeFilter_guided_gpu(gpu_volume, l_im, ksize, c_sigma);
	//costVolumeFilter_box_gpu(gpu_volume, ksize);
	costVolumeMinimize_gpu(gpumat_volume, out_gpu, 64);

	//struct cost_volume_t cpu_volume = get_gpu_volume(gpu_volume);
	//costVolumeFilter_guided(cpu_volume,l_im,ksize,c_sigma);
	//costVolumeMinimize(cpu_volume,out_gpu);

	//Mat refmat_volume = createCostVolume(l_im,r_im,64);
	Mat refmat_volume = createCostVolume_tadcg(l_im,r_im,64,20,90,.9);
	struct cost_volume_t ref_volume = cost_volume_from_mat(refmat_volume,64);
	//costVolumeFilter_jointBilateral(ref_volume, l_im, ksize, c_sigma, s_sigma);
	//costVolumeBoxFilter(ref_volume,ksize);
	//costVolumeFilter_guided(ref_volume,l_im,ksize,c_sigma);
	costVolumeMinimize(ref_volume, out);
	//viewSlices(cpu_volume,0,12);
	//viewSlices(ref_volume,0,10);
	int show = 1;
	if(show){
		printf("l_im\n"); imshow("window",l_im); waitKey(0);
		printf("cpu\n");  imshow("window",out); waitKey(0);
		printf("gpu\n");  imshow("window",out_gpu); waitKey(0);
		printf("python\n");  imshow("window",imread("tadcg.png")); waitKey(0);
	}
	int write = 1;
	if(write){
		imwrite("l_im",l_im);
		imwrite("out",out);
		imwrite("out_gpu",out_gpu);
	}
}

