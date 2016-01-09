#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>


#define MAX_BLOCK_SIZE 32
#define MAX_WINDOW_SIZE 55
#define MAX_DISP 1000

using namespace std;
using namespace cv;

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

// Here's the plan!
// I need to program the kernel to work with a no-boundaries condition
// Then I can modify it to deal with weird boundaries or large disparities
// And also it will need to have a check for very large 


// In the future it may be useful to bring a whole line of pixels into local memory...
// ... from shared memory, and then do everything that needs to be dones with that line...
// ... for a given pixel, before moving to the next row...
// ... or maybe it would be better to use a single location of spacial sigma.  Oh I like that. 

// Device code
__global__ void asw_kernel(unsigned char* global_left, unsigned char* global_right, unsigned char* output,
	int nrows, int ncols, int nchans, int ndisp, int win_size, int win_rad, float s_sigma, float c_sigma)
	{
	// ok, we're going to try a block size of 32 ( 32x32 = 1024, max threads per block )
	// no... we'll use 16x16 since there's problems with shared memory with two images
	// each thread will calculate the full asw stereo output for a single pixel
	// shared memory will contain all the input image data for the full block of asw calculations
	// texture memory will contain the spacial filter, eventually
	extern __shared__ char im[]; // contains both left and right image data

	// get the size of the sub-image that we are considering
	int im_width_pix = ndisp+win_size+blockDim.x;
	int im_width_bytes = im_width_pix*3*sizeof(char);
	int im_rows = win_size+blockDim.y;

	char* im_l = im; // reference to the beginnning of im[]
	char* im_r = (char*)(&im[ im_width_bytes*im_rows ]); // reference to the middleish of im[]
	float c_factor;  // this will have to be recalculated every time
	float s_factor; // I think I can calculate this each time for now
	float weights[MAX_DISP];
	float values[MAX_DISP];
	float center_value;
	float val;
	char temp;

	// get identity of this thread
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// setup LUTs // nevermind... right now there are none

	// copy relevant sections of image to shared memory
	// TODO : additional boundary checks on this data
	// we can copy the 24-bit image over 32 bits at a time
	// except then I don't know how to deal with the edge case
	// so let's just do one character at a time
	for(int i = 0; blockDim.x*i < im_width_bytes; i++){
		int x_idx = blockDim.x*i + tx;
		// for testing, we will have one sub_image that just fits nicely in the corner
		int global_x_idx = x_idx;
		if(x_idx < im_width_bytes){
			for(int j = 0; blockDim.y*j < im_rows; j++){
				int y_idx = blockDim.y;
				// for testing, we will have one sub_image that just fits nicely in the corner
				int global_y_idx = y_idx;
				if(y_idx < im_rows){
					// copy pixels from each global_images into the sub_images
					im_l[y_idx*im_width_bytes + x_idx] = global_left[global_y_idx*ncols*nchans + global_x_idx];
					im_r[y_idx*im_width_bytes + x_idx] = global_right[global_y_idx*ncols*nchans + global_x_idx];
				}
			}
		}
	}

	__syncthreads();

	// copy directly to output, just so we can do some testing.
	for(int i = 0; blockDim.x*i < im_width_bytes; i++){
		int x_idx = blockDim.x*i + tx;
		// for testing, we will have one sub_image that just fits nicely in the corner
		int global_x_idx = x_idx;
		if(x_idx < im_width_bytes){
			for(int j = 0; blockDim.y*j < im_rows; j++){
				int y_idx = blockDim.y;
				// for testing, we will have one sub_image that just fits nicely in the corner
				int global_y_idx = y_idx;
				if(y_idx < im_rows){
					// copy pixels from each global_images into the sub_images
					output[global_y_idx*ncols*nchans + global_x_idx] = im_l[y_idx*im_width_bytes + x_idx];
				}
			}
		}
	}

	__syncthreads();


	// in each row of the window:
		// in each column of the window:
			// for each pixel in the window

}

int asw(Mat im_l, Mat im_r, int ndisp, int s_sigma, int i_sigma){
	// window size and win_rad
	int win_size = 3*s_sigma-1;
	int win_rad = (win_size - 1)/2;

	// check that images are matching dimensions
	if(im_l.rows != im_r.rows){
		printf("Error: im_l and im_r do not have matching row count\n");
		return 1;
	}
	if(im_l.cols != im_r.cols){
		printf("Error: im_l and im_r do not have matching col count\n");
		return 1;
	}
	if(im_l.channels() != im_r.channels()){
		printf("Error: im_l and im_r do not have matching channel count\n");
		return 1;
	}

	// set easy-access variables for number of rows, cols, and chans
	int nrows = im_l.rows;
	int ncols = im_l.cols;
	int nchans = im_l.channels();
	// initialize the device input arrays
	unsigned char* d_im_l;
	cudaMalloc(&d_im_l,nchans*nrows*ncols*sizeof(unsigned char));
	unsigned char* d_im_r;
	cudaMalloc(&d_im_r,nchans*nrows*ncols*sizeof(unsigned char));
	// initialize the output data matrix
	unsigned char* out = (unsigned char*)malloc(nrows*ncols*sizeof(unsigned char));
	unsigned char* d_out;
	cudaMalloc(&d_out,nrows*ncols*sizeof(unsigned char));

	// define a shortcut to the host data arrays
	unsigned char* data_l = ((unsigned char*)(im_l.data));
	unsigned char* data_r = ((unsigned char*)(im_r.data));

	//copy the host input data to the device
    cudaMemcpy(d_im_l, data_l, nchans*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im_r, data_r, nchans*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

	// get gaussian kernel for spacial look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	float s_weights[win_size][win_size]; 
	for(int i=0; i<win_size; i++){
		for(int j=0; j<win_size; j++){
			float x = i-win_rad;
			float y = j-win_rad;
			float radius = sqrt(x*x+y*y);
			s_weights[i][j] = std::pow(2.71828,-radius*radius/(2.*s_sigma*s_sigma));
			// printf("%.6f ",s_weights[i][j]);
		}
		// printf("\n");
	}

	// get gaussian kernel for intensity look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	float i_weights[511]; 
	for(int i=0; i<511; i++){
		float radius = i-255;
		i_weights[i] = std::pow(2.71828,-radius*radius/(2.*i_sigma*i_sigma));
		// printf("%.6f ",i_weights[i]);
	}

	// call the kernel here!

	// copy the host input data to the device
    cudaMemcpy(out, d_out, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// cleanup memory
	cudaFree(d_im_l);
	cudaFree(d_im_r);
	cudaFree(d_out);
	free(out);

	return 0;
}

int main(int argc, char** argv){
	// spacial and intensity sigmas
	int s_sigma, i_sigma;
	// number of disparities to check
	int ndisp;
	// input images
	Mat im_l, im_r;

	if(argc < 6){
		printf("usage: %s <left image> <right image> <num disparities> <spacial sigma> <color sigma>",argv[0]);
		return 1;
	}else{
		im_l = imread(argv[1]);
		im_r = imread(argv[2]);
		ndisp = atoi(argv[3]);
		s_sigma = atoi(argv[4]);
		i_sigma = atoi(argv[5]);
	}

	return asw(im_l, im_r, ndisp, s_sigma, i_sigma);
}