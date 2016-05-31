#include "costVolumeFilter_guided.h"
#include "helper.h"
#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include <npp.h>


using namespace std;
using namespace cv;

__global__ void costVolumeFilter_guided_kernel(struct cost_volume_t vol, int* guide_global, int inter_win_padding, float* output, float sigma_s, float sigma_c, int ksize){
	
}

void costVolumeFilter_guided_gpu(struct cost_volume_t& vol, Mat guide, int ksize, float eps){
	int nrows = vol.nrows;
	int ncols = vol.ncols;
	int ndisp = vol.ndisp;
	int stride = vol.stride;

	struct timespec timer;

	cuda::GpuMat I;

	// copy guide image to grayscale
	cvtColor(guide,guide,CV_BGR2GRAY);
	// convert to float
	guide.convertTo(guide,CV_32FC1);
	// copy guide image to GPU
	I.upload(guide);
	// set up working memory
	cuda::GpuMat mean(I.rows,I.cols,I.type());
	cuda::GpuMat var(I.rows,I.cols,I.type());
	cuda::GpuMat workmem(I.rows,I.cols,I.type());
	cuda::GpuMat workmem2(I.rows,I.cols,I.type());
	cuda::GpuMat workmem3(I.rows,I.cols,I.type());

	cuda::GpuMat p_(I.rows,I.cols,I.type());
	cuda::GpuMat p_mean(I.rows,I.cols,I.type());
	cuda::GpuMat a(I.rows,I.cols,I.type());
	cuda::GpuMat a_(I.rows,I.cols,I.type());
	cuda::GpuMat a_mean(I.rows,I.cols,I.type());
	cuda::GpuMat a_I(I.rows,I.cols,I.type());
	cuda::GpuMat b(I.rows,I.cols,I.type());
	cuda::GpuMat b_(I.rows,I.cols,I.type());
	cuda::GpuMat Ip(I.rows,I.cols,I.type());
	cuda::GpuMat Ip_(I.rows,I.cols,I.type());

	check_timer(NULL,&timer);

	// pre-step 1: box filter I to get mean
	cudaDeviceSynchronize();
	{
		float* src_data  = (float*)I.data;
		float* out_data  = (float*)mean.data;
		int src_pitch    = I.step;
		int out_pitch    = mean.step;
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
	// pre-step 2: square I, for variance calculation
	cudaDeviceSynchronize();
	cuda::sqr(I,var);
	// pre-step 3: box filter I^2
	cudaDeviceSynchronize();
	{
		float* src_data  = (float*)var.data;
		float* out_data  = (float*)workmem3.data;
		int src_pitch    = var.step;
		int out_pitch    = workmem3.step;
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
	// pre-step 4: square the mean
	cudaDeviceSynchronize();
	cuda::sqr(mean,workmem);
	// pre-step 5: variance = mean(x^2) - mean(x)^2
	cudaDeviceSynchronize();
	cuda::subtract(workmem3, workmem, workmem2);
	// pre-step 6: add eps to variance
	cudaDeviceSynchronize();
	cuda::add(workmem2, eps, var);

	for(int disp = 0; disp < ndisp; disp++){
		// step 1: element-wise multiply I by p
		cuda::GpuMat p(Size(ncols,nrows), CV_32F, &(vol.volume[disp*nrows*stride]), stride*sizeof(float));
		//cuda::GpuMat Ip = workmem;
		cuda::multiply(I,p,Ip);
		// step 2: box filter Ip to be Ip_
		//cuda::GpuMat Ip_ = Ip;
		{
			float* src_data  = (float*)Ip.data;
			float* out_data  = (float*)Ip_.data;
			int src_pitch    = Ip.step;
			int out_pitch    = Ip_.step;
			NppiSize size    = {ncols, nrows };
			NppiSize sizeROI = {ncols, nrows };
			NppiSize kernel  = {ksize , ksize };
			NppiPoint offset = {0 , 0 };
			NppiPoint anchor = {ksize/2 , ksize/2 };

			nppiFilterBoxBorder_32f_C1R(
				src_data, src_pitch,
				size, offset,
				out_data, out_pitch,
				sizeROI, kernel, anchor, NPP_BORDER_REPLICATE);
		}
		// step 3: box filter p to be p_
		//cuda::GpuMat p_ = p;
		{
			float* src_data  = (float*)p.data;
			float* out_data  = (float*)p_.data;
			int src_pitch    = p.step;
			int out_pitch    = p_.step;
			NppiSize size    = {ncols, nrows };
			NppiSize sizeROI = {ncols, nrows };
			NppiSize kernel  = {ksize , ksize };
			NppiPoint offset = {0 , 0 };
			NppiPoint anchor = {ksize/2 , ksize/2 };

			nppiFilterBoxBorder_32f_C1R(
				src_data, src_pitch,
				size, offset,
				out_data, out_pitch,
				sizeROI, kernel, anchor, NPP_BORDER_REPLICATE);
		}
		// step 4: combine p_ and mean
		//cuda::GpuMat p_mean = workmem2;
		cuda::multiply(p_, mean, p_mean);
		// step 5: compute Ip_ - mean*p_
		//cuda::GpuMat a = Ip_;
		cuda::subtract(Ip_, p_mean, workmem);
		// step 6: divide by var+eps (stored as var)
		cuda::divide(workmem, var, a);
		// step 7: start calculating b with a*mean
		//cuda::GpuMat a_mean  = workmem2;
		cuda::multiply(a, mean, a_mean);
		// step 8: b = p_ - a_mean
		//cuda::GpuMat b = p_;
		cuda::subtract(p_, a_mean, b);
		// step 9: box filter a
		//cuda::GpuMat a_ = a;
		{
			float* src_data  = (float*)a.data;
			float* out_data  = (float*)a_.data;
			int src_pitch    = a.step;
			int out_pitch    = a_.step;
			NppiSize size    = {ncols, nrows };
			NppiSize sizeROI = {ncols, nrows };
			NppiSize kernel  = {ksize , ksize };
			NppiPoint offset = {0 , 0 };
			NppiPoint anchor = {ksize/2 , ksize/2 };

			nppiFilterBoxBorder_32f_C1R(
				src_data, src_pitch,
				size, offset,
				out_data, out_pitch,
				sizeROI, kernel, anchor, NPP_BORDER_REPLICATE);
		}
		// step 10: box filter b
		//cuda::GpuMat b_ = b;
		{
			float* src_data  = (float*)b.data;
			float* out_data  = (float*)b_.data;
			int src_pitch    = b.step;
			int out_pitch    = b_.step;
			NppiSize size    = {ncols, nrows };
			NppiSize sizeROI = {ncols, nrows };
			NppiSize kernel  = {ksize , ksize };
			NppiPoint offset = {0 , 0 };
			NppiPoint anchor = {ksize/2 , ksize/2 };

			nppiFilterBoxBorder_32f_C1R(
				src_data, src_pitch,
				size, offset,
				out_data, out_pitch,
				sizeROI, kernel, anchor, NPP_BORDER_REPLICATE);
		}
		// step 11: start to build q with a_ * I
		//cuda::GpuMat a_I = a_;
		cuda::multiply(a_, I, a_I);
		// step 12: q = a_I + b_;
		cuda::GpuMat q = p;
		cuda::add(a_I, b_, q);
	}

	check_timer("costVolumeFilter_guided_gpu time",&timer);

	I.release();
	mean.release();
	var.release();
	workmem.release();
}

void costVolumeFilter_guided(struct cost_volume_t& vol, Mat guide, int ksize, float eps){
	int nrows = vol.nrows;
	int ncols = vol.ncols;
	int ndisp = vol.ndisp;
	float* vin = vol.volume;
	// doesn't do in-place editing... need second float*
	float* vout = (float*)malloc(nrows*ncols*ndisp*sizeof(float));
	// create guided filter
	//Ptr<ximgproc::GuidedFilter> guided = ximgproc::createGuidedFilter(guide,ksize,eps);

	cvtColor(guide,guide,CV_BGR2GRAY);

	struct timespec timer;
	check_timer(NULL,&timer);

	for(int disp = 0; disp < ndisp; disp++){
		Rect relevant;
		relevant.x = disp; relevant.width = ncols-disp;
		relevant.y = 0; relevant.height = nrows;
		Mat slicein(nrows,ncols,CV_32F,&(vin[nrows*ncols*disp]));
		Mat sliceout(nrows,ncols,CV_32F,&(vout[nrows*ncols*disp]));
		ximgproc::guidedFilter(guide(relevant),slicein(relevant),sliceout(relevant),ksize,eps);
	}

	check_timer("costVolumeFilter_guided time:",&timer);
	printf("\n");
	// free old cost_volume float*
	free(vol.volume);
	// replace with new cost_volume float*
	vol.volume = vout;
}

