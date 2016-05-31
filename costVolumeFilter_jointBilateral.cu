#include "costVolumeFilter_jointBilateral.h"
#include "opencv2/ximgproc/edge_filter.hpp"
#include "helper.h"

using namespace std;
using namespace cv;

__global__ void costVolumeFilter_jointBilateral_kernel(struct cost_volume_t vol, int* guide_global, int inter_win_padding, float* output, int ksize, float sigma_c, float sigma_s){
	int gx = blockIdx.x*blockDim.x + threadIdx.x;
	int gy = blockIdx.y*blockDim.y + threadIdx.y;

	// radius of kernel
	int krad = (ksize-1)/2;

	extern __shared__ char shared_mem[];

	// the guide is first in shared memory
	int* guide = (int*)(&shared_mem[0]);
	// the slice is second in shared memory
	float* slice = (float*)&(shared_mem[(ksize+blockDim.y-1)*(ksize+blockDim.x-1)*sizeof(float) + inter_win_padding]);

	int guide_c;

	// center pixel of guide image
	if(gy < vol.nrows && gx < vol.ncols){
		guide_c = guide_global[vol.ncols*gy + gx];
	}
	// pull out channel data from guide center pixel (brought in as an int)
	int gcr,gcg, gcb;
	gcr = (guide_c&0x000000FF) >> 0;
	gcb = (guide_c&0x0000FF00) >> 8;
	gcg = (guide_c&0x00FF0000) >> 16;

	// copy relevant subimages to shared memory
	// starting with the guide sub image
	for(int i = 1; i < ksize+blockDim.x-1; i += blockDim.x){
		// only threads in bounds in x dim continue to next loop
		if(i + threadIdx.x < ksize+blockDim.x-1 && gx + i - krad >= 0 && gx + i - krad < vol.ncols){
			for(int j = 0; j < ksize+blockDim.y-1; j += blockDim.y){
				// only threads in bounds in y dim continue
				if(j + threadIdx.y < ksize+blockDim.y-1 && gy + j - krad >= 0 && gy + j - krad < vol.nrows){
					guide[(ksize+blockDim.x-1) * (j+threadIdx.y) + i + threadIdx.x] = guide_global[vol.ncols * (gy + j - krad) + gx + i - krad];
				}
			}
		}
	}
	__syncthreads();
	// continuing with the slice sub image
	for(int i = 0; i < ksize+blockDim.x-1; i += blockDim.x){
		// only threads in bounds in x dim continue to next loop
		if(i + threadIdx.x < ksize+blockDim.x-1 && gx + i - krad >= 0 && gx + i - krad < vol.ncols){
			for(int j = 0; j < ksize+blockDim.y-1; j += blockDim.y){
				// only threads in bounds in y dim continue
				if(j + threadIdx.y < ksize+blockDim.y-1 && gy + j - krad >= 0 && gy + j - krad < vol.nrows){
					slice[(ksize+blockDim.x-1) * (j+threadIdx.y) + i + threadIdx.x] = vol.volume[vol.nrows*vol.stride*blockIdx.z + vol.stride * (gy + j - krad) + gx + i - krad];
				}
			}
		}
	}
	__syncthreads();

	float weight = 0;
	float sum = 0;

	// now the bilateral calculation
	for(int i = 0; i < ksize; i++){
		if(gx - krad + i >= 0 && gx - krad + i < vol.ncols){
			for(int j = 0; j < ksize; j++){
				if(gy - krad + j >= 0 && gy - krad + j < vol.nrows){
					int   guide_p  = guide[(ksize+blockDim.x-1)*(j+threadIdx.y) + i + threadIdx.x];
					float slice_p  = slice[(ksize+blockDim.x-1)*(j+threadIdx.y) + i + threadIdx.x];
					int gr,gg,gb;
					gr = (guide_p&0x000000FF) >> 0;
					gb = (guide_p&0x0000FF00) >> 8;
					gg = (guide_p&0x00FF0000) >> 16;
					int c_diff = abs(gr - gcr) + abs(gb - gcb) + abs(gg - gcg);
					float s = __expf( -((j-krad)*(j-krad)+(i-krad)*(i-krad)) / (sigma_s*sigma_s) );
					float c = __expf( -(c_diff*c_diff) / (sigma_c*sigma_c));
					weight += s*c;
					sum += slice_p*s*c;
				}
			}
		}
		__syncthreads();
	}

	// normalize the weighted sum by the sum of the weights
	sum /= weight;

	if(gy < vol.nrows && gx < vol.ncols){
		// for debug, just copy the guide sub image to the output buffer
		//output[vol.nrows*vol.stride*blockIdx.z + vol.stride*gy + gx] = (float)(guide[(ksize+blockDim.x-1)*(threadIdx.y + krad) + krad + threadIdx.x] & 0x000000FF);
		// for debug, just copy the slice sub image to the output buffer
		//output[vol.nrows*vol.stride*blockIdx.z + vol.stride*gy + gx] = slice[(ksize+blockDim.x-1)*(threadIdx.y + krad) + krad + threadIdx.x];
		// ok but for reals, output the bilaterally smoothed value here
		output[vol.nrows*vol.stride*blockIdx.z + vol.stride*gy + gx] = sum;
	}
}

void costVolumeFilter_jointBilateral_gpu(struct cost_volume_t& cost_volume, Mat guide, int ksize, float sigma_c, float sigma_s){
	int nrows = cost_volume.nrows;
	int ncols = cost_volume.ncols;
	int ndisp = cost_volume.ndisp;
	int stride = cost_volume.stride;

	if(ksize%2 != 1){
		printf("ERROR: in costVolumeFilter_jointBilateral_gpu, ksize must be odd\n");
		return;
	}

	// settings for the kernel
	// trying to use 32 threads-wide so the global reads are 128 bytes
	dim3 threadsPerBlock(32,16);
	dim3 blocksPerGrid(ncols/threadsPerBlock.x+1,nrows/threadsPerBlock.y+1,ndisp);
	int guide_win_rows = (ksize + threadsPerBlock.y - 1);
	int guide_win_width_bytes = (ksize + threadsPerBlock.x - 1)*sizeof(int);
	// pad between images to 256 bytes
	int inter_window_pad = (256 - guide_win_width_bytes%256)%256;
	int slice_win_rows = (ksize + threadsPerBlock.y - 1);
	int slice_win_width_bytes = (ksize + threadsPerBlock.x - 1)*sizeof(float);
	int shared_size = guide_win_rows*guide_win_width_bytes + inter_window_pad + slice_win_rows*slice_win_width_bytes;
	// make sure the shared size is less than device maximum
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	if(shared_size > properties.sharedMemPerMultiprocessor){
		printf("ERROR: in costVolumeFilter_jointBilateral_gpu, shared_size exceeds device limit\n");
		return;
	}

	// allocate output volume (post-filtering) on gpu
	float* d_output;
	cudaMalloc(&d_output, ndisp*nrows*stride*sizeof(float));
	// copy guide image to to GPU
	cvtColor(guide,guide,CV_BGR2RGBA);
	int* d_guide;
	cudaMalloc(&d_guide, 4*nrows*ncols*sizeof(unsigned char));
    cudaMemcpy(d_guide, guide.data, 4*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	// call the kernel
	struct timespec timer;
	check_timer(NULL,&timer);
    costVolumeFilter_jointBilateral_kernel<<<blocksPerGrid, threadsPerBlock, shared_size>>>(cost_volume, d_guide, inter_window_pad, d_output, ksize, sigma_c, sigma_s);
    //costVolumeFilter_jointBilateral_kernel<<<blocksPerGrid, threadsPerBlock>>>(cost_volume, d_guide, inter_window_pad, d_output, sigma_s, sigma_c, ksize);
	cudaDeviceSynchronize();
    check_timer("costVolumeFilter_jointBilateral_gpu time",&timer);
	gpu_perror("costVolumeFilter_jointBilateral_kernel");

	// shuffle cost_volume pointers
	cudaFree(cost_volume.volume); // don't need the input anymore
	cost_volume.volume = d_output; // keep the output instead
}

void jointBilateralFilter(Mat& srcim, Mat& guideim, Mat& dst, int kernelSize, float sigma_color, float sigma_space){
	// make sure images are the same size
	if(srcim.cols != guideim.cols || srcim.rows != guideim.rows){
		printf("ERROR: src and guide images in jointBilateralFilter do not have matching rows and cols\n");
		return;
	}
	if(kernelSize%2 != 1){
		printf("ERROR: kernelSize jointBilateralFilter must be odd\n");
		return;
	}
	int nrows = srcim.rows;
	int ncols = srcim.cols;
	int nchans = guideim.channels();
	// set up some useful variables
	int win_rad = (kernelSize -1) / 2;
	// assume we are taking in floating point images
	float* src = (float*)srcim.data;
	float* guide = (float*)guideim.data;
	Mat outim = Mat::zeros(nrows,ncols,CV_32F);
	float* out = (float*)outim.data;
	// iterate over the whole image
	for(int col = 0; col < ncols; col++){
		for(int row = 0; row < nrows; row++){
			double normalizing_factor = 0;
			double weighted_sum = 0;
			float* guide_center = &(guide[(ncols*row + col)*nchans]);
			// iterate over the window
			for(int j = max(0,row-win_rad); j < min(nrows,row+win_rad+1); j++){
				for(int i = max(0,col-win_rad); i < min(ncols,col+win_rad+1); i++){
					int x = i - col;
					int y = j - row;
					int radius2 = x*x+y*y;
					float src_pixel = src[ncols*j + i];
					float* guide_pixel = &(guide[(ncols*j + i)*nchans]);
					double weight = 1;
					// apply spacial sigma
					weight *= std::exp(-radius2/(2.*sigma_space*sigma_space));
					// get intensity difference from guide image
					float diff = 0;
					for(int chan = 0; chan < nchans; chan++){
						diff += abs(guide_pixel[chan] - guide_center[chan]);
					}
					// apply sigma_color
					weight *= std::exp(-diff*diff/(2.*sigma_color*sigma_color));
					// add in values
					normalizing_factor += weight;
					weighted_sum += weight*src_pixel;
				}
			}
			out[ncols*row + col] = weighted_sum / normalizing_factor;
			//printf("row,col,val : %d,%d,%f\n",row,col,weighted_sum / normalizing_factor);
		}
	}
	outim.copyTo(dst);
	return;
}

void costVolumeFilter_jointBilateral(struct cost_volume_t& cost_volume, Mat guide, int kernelSize, float sigma_color, float sigma_space){
	int nrows = cost_volume.nrows;
	int ncols = cost_volume.ncols;
	int ndisp = cost_volume.ndisp;
	float* vin = cost_volume.volume;
	// doesn't do in-place editing... need second float*
	float* vout = (float*)malloc(nrows*ncols*ndisp*sizeof(float));
	// guide must be CV_32F if the cost_volume is
	guide.convertTo(guide,CV_32F);
	struct timespec timer;
	check_timer(NULL,&timer);
	for(int disp = 0; disp < ndisp; disp++){
		Mat slicein(nrows,ncols,CV_32F,&(vin[nrows*ncols*disp]));
		Mat sliceout(nrows,ncols,CV_32F,&(vout[nrows*ncols*disp]));
		//jointBilateralFilter(InputArray joint, InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT)
		ximgproc::jointBilateralFilter(guide, slicein, sliceout, kernelSize, sigma_color, sigma_space);
	}
	check_timer("costVolumeFilter_jointBilateral time",&timer);
	printf("\n");
	// free old cost_volume float*
	free(cost_volume.volume);
	// replace with new cost_volume float*
	cost_volume.volume = vout;
}
