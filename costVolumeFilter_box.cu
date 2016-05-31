#include "costVolumeFilter_box.h"
#include "helper.h"
#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/cudafilters.hpp"
#include <npp.h>

using namespace std;
using namespace cv;

void costVolumeFilter_box_gpu(struct cost_volume_t& vol, int ksize){
	int nrows = vol.nrows;
	int ncols = vol.ncols;
	int ndisp = vol.ndisp;
	int stride = vol.stride;

	// output volume
	float* d_output;
	cudaMalloc(&d_output, ndisp*nrows*stride*sizeof(float));

	struct timespec timer;
	check_timer(NULL,&timer);

	for(int disp = 0; disp < ndisp; disp++){
		float* src_data  = &(vol.volume[disp*nrows*stride]);
		float* out_data  = &(d_output[disp*nrows*stride]);
		int src_pitch    = stride*sizeof(float);
		int out_pitch    = stride*sizeof(float);
		NppiSize size    = {ncols , nrows };
		NppiSize sizeROI = {ncols , nrows };
		NppiSize kernel  = {ksize , ksize };
		NppiPoint offset = {0 , 0 };
		NppiPoint anchor = {ksize/2 , ksize/2 };


		nppiFilterBoxBorder_32f_C1R(
			src_data, src_pitch,
			size, offset,
			out_data, out_pitch,
			sizeROI, kernel, anchor, NPP_BORDER_REPLICATE);
	}

	check_timer("costVolumeFilter_box_gpu time",&timer);

	// shuffle pointers
	cudaFree(vol.volume);
	vol.volume = d_output;
}


void costVolumeFilter_box(struct cost_volume_t& cost_volume, int kernelSize){
	int nrows = cost_volume.nrows;
	int ncols = cost_volume.ncols;
	int ndisp = cost_volume.ndisp;
	float* vin = cost_volume.volume;
	// doesn't do in-place editing... need second float*
	float* vout = (float*)malloc(nrows*ncols*ndisp*sizeof(float));

	struct timespec timer;
	check_timer(NULL,&timer);

	for(int disp = 0; disp < ndisp; disp++){
		Mat slicein(nrows,ncols,CV_32F,&(vin[nrows*ncols*disp]));
		Mat sliceout(nrows,ncols,CV_32F,&(vout[nrows*ncols*disp]));
		boxFilter(slicein, sliceout, -1, Size(kernelSize,kernelSize));
	}

	check_timer("costVolumeFilter_box time",&timer);

	// free old cost_volume float*
	free(cost_volume.volume);
	// replace with new cost_volume float*
	cost_volume.volume = vout;
}

