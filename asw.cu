#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>


#define MAX_BLOCK_SIZE 32
#define MAX_WINDOW_SIZE 55
#define MAX_DISP 1000

#define BLOCK_SIZE 16

// timing utility
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

// little bitty kernel to initialize blocks of device memory
__global__ void gpu_memset(unsigned char* start, unsigned char value, int length){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int gx = bx*blockDim.x + tx;
	if(gx < length){
		start[gx] = value;
	}
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
__global__ void asw_kernel(unsigned char* global_left, unsigned char* global_right, unsigned char* output, unsigned char* debug,
	int nrows, int ncols, int nchans, int ndisp, int win_size, int win_rad, float s_sigma, float c_sigma)
	{
	// ok, we're going to try a block size of 32 ( 32x32 = 1024, max threads per block )
	// no... we'll use 16x16 since there's problems with shared memory with two images
	// each thread will calculate the full asw stereo output for a single pixel
	// shared memory will contain all the input image data for the full block of asw calculations
	// texture memory will contain the spacial filter, eventually
	extern __shared__ unsigned char im[]; // contains both left and right image data

	// get the size of the sub-images that we are considering
	// reference window
	int ref_width_pix = 2*win_rad+blockDim.x;
	int ref_width_bytes = ref_width_pix*nchans*sizeof(unsigned char);
	int ref_rows = 2*win_rad+blockDim.y;
	// target window
	int tgt_width_pix = ndisp+2*win_rad+blockDim.x;
	int tgt_width_bytes = tgt_width_pix*nchans*sizeof(unsigned char);
	int tgt_rows = 2*win_rad+blockDim.y;

	unsigned char* ref = im; // left image, reference to the beginnning of im[]
	unsigned char* tgt = (unsigned char*)(&im[ ref_width_bytes*ref_rows ]); // right image, reference to somwhere in middle of im[]

	float ref_c_factor;  // this will have to be recalculated every time
	float tgt_c_factor;  // this will have to be recalculated every time
	float s_factor; // I think I can calculate this each time for now
	float weight;
	float weights[MAX_DISP];
	float cost;
	float costs[MAX_DISP];
	unsigned char* ref_center_pix;
	unsigned char* ref_pix;
	unsigned char* tgt_center_pix;
	unsigned char* tgt_pix;

	// get identity of this thread
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x + 15;
	int by = blockIdx.y + 15;
	int gx = bx*blockDim.x + tx;
	int gy = by*blockDim.y + ty;

	// setup LUTs // nevermind... right now there are none

	// copy relevant subimages to shared memory
	// TODO: additional boundary checks on this data
	// TODO: better division technique
	// TODO: investigate where syncthreads() needs to be called for best performance
	// we can copy the 24-bit image over 32 bits at a time
	// except then I don't know how to deal with the edge case
	// so let's just do one character at a time
	// starting with reference image:
	int xblocks = ref_width_bytes / blockDim.x + 1;
	int yblocks = ref_rows / blockDim.y + 1;
	int xstart = (bx*blockDim.x - win_rad)*nchans;
	int ystart = gy - win_rad;
	for(int i = 0; i < xblocks; i++){
		int x_idx = i*blockDim.x + tx;
		int g_x_idx = xstart + i*blockDim.x + tx;
		if(x_idx < ref_width_bytes){
			for(int j = 0; j < yblocks; j++){
				int y_idx = j*blockDim.y + ty;
				int g_y_idx = ystart + j*blockDim.y;
				if(y_idx < ref_rows){
					// copy bytes (not pixels) from global_left into reference image
					ref[y_idx*ref_width_bytes + x_idx] = global_left[g_y_idx*ncols*nchans + g_x_idx];
					// copy into the debug image (only made to work with a single block of threads)
					// debug[g_y_idx*ncols*nchans + g_x_idx]  = ref[y_idx*ref_width_bytes + x_idx];
				}
			}
		}
	}
	// then to the target image:
	xblocks = tgt_width_bytes / blockDim.x + 1;
	yblocks = tgt_rows / blockDim.y + 1;
	xstart = (bx*blockDim.x - win_rad - ndisp)*nchans;
	ystart = gy - win_rad;
	for(int i = 0; i < xblocks; i++){
		int x_idx = i*blockDim.x + tx;
		int g_x_idx = xstart + i*blockDim.x + tx;
		if(x_idx < tgt_width_bytes){
			for(int j = 0; j < yblocks; j++){
				int y_idx = j*blockDim.y + ty;
				int g_y_idx = ystart + j*blockDim.y;
				if(y_idx < tgt_rows){
					// copy bytes (not pixels) from global_left into reference image
					tgt[y_idx*tgt_width_bytes + x_idx] = global_right[g_y_idx*ncols*nchans + g_x_idx];
					// copy into the debug image (only made to work with a single block of threads)
					debug[g_y_idx*ncols*nchans + g_x_idx]  = tgt[y_idx*tgt_width_bytes + x_idx];
				}
			}
		}
	}

	__syncthreads();

	// initialize weights and costs
	for(int i = 0; i < MAX_DISP; i++){
		costs[i] = 0;
		weights[i] = 0;
	}

	// get a pointer to the ref_center_pix, which is constant for any given thread
	ref_center_pix = &ref[(win_rad + ty)*ref_width_bytes + (win_rad + tx)*nchans];

	// in each row in the window:
	for(int win_x = 0; win_x < win_size; win_x++){
		// locate the pixel in the ref image
		int ref_x = win_x + tx;
		// find the window-center to pixel x-distance
		int dx = win_x - win_rad;
		// in each column of the window:
		for(int win_y = 0; win_y < win_size; win_y++){
			// locate the pixel in the ref image
			int ref_y = win_y + ty;
			// get a pointer to the pixel
			ref_pix = &ref[ref_y*ref_width_bytes + ref_x*nchans];
			// get the ref center-to-pixel color difference
			float ref_c2p_diff = abs(ref_center_pix[0] - ref_pix[0]);
			// include additional channels
			for(int i = 1; i < nchans; i++){
				ref_c2p_diff += abs(ref_center_pix[i] - ref_pix[i]);
			}
			// get the ref_c_factor
			ref_c_factor = pow(2.71828,-ref_c2p_diff*ref_c2p_diff/(2.*c_sigma*c_sigma));
			// find the window-center to pixel y-distance
			int dy = win_y - win_rad;
			float radius_2 = dx*dx + dy*dy;
			// get the s_factor for this particular window location
			s_factor = pow(2.71828,-radius_2/(2.*s_sigma*s_sigma));
			// for each value of ndisp:
			for(int disp = 0; disp < ndisp; disp++){
				// get a pointer to the tgt_center_pix, which changes for each disp
				tgt_center_pix = &tgt[(win_rad + ty)*tgt_width_bytes + (ndisp + win_rad + tx - disp)*nchans];
				// locate the pixel in the tgt image
				int tgt_x = ndisp + win_x + tx - disp;
				int tgt_y = ref_y;
				// get a pointer to the pixel
				tgt_pix = &tgt[tgt_y*tgt_width_bytes + tgt_x*nchans];
				// get the tgt center-to-pixel color difference
				float tgt_c2p_diff = abs(tgt_center_pix[0] - ref_pix[0]);
				// include additional channels
				for(int i = 1; i < nchans; i++){
					tgt_c2p_diff += abs(tgt_center_pix[i] - ref_pix[i]);
				}
				// get the tgt_c_factor
				tgt_c_factor = pow(2.71828,-tgt_c2p_diff*tgt_c2p_diff/(2.*c_sigma*c_sigma));
				// get the ref2tgt_diff
				float ref2tgt_diff = abs(ref_pix[0] - tgt_pix[0]);
				// include additional channels
				for(int i = 1; i < nchans; i++){
					ref2tgt_diff+= abs(ref_pix[i] - tgt_pix[i]);
					ref2tgt_diff+= abs(ref_pix[i] - tgt_pix[i]);
				}
				//calulate the weight
				weight = s_factor*ref_c_factor*tgt_c_factor;
				// add in the cost
				costs[disp] += weight*ref2tgt_diff;
				// add in the weight
				weights[disp] += weight;
			}
		}
	}

	__syncthreads();

	// now go through and find the lowest normalized cost
	float min_cost = costs[0]/weights[0];
	int index = 0;
	for(int disp = 1; disp < ndisp; disp++){
		cost = costs[disp]/weights[disp];	
		if(cost < min_cost){
			min_cost = cost;
			index = disp;
		}
		__syncthreads();
	}
	// set the output to the index of min_cost
	output[gy*ncols + gx] = index;
}

int asw(cv::Mat im_l, cv::Mat im_r, int ndisp, int s_sigma, int c_sigma){
	// window size and win_rad
	int win_size = 3*s_sigma-1;
	int win_rad = (win_size - 1)/2;
	// declare timer
	struct timespec timer;

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
	unsigned char* debug = (unsigned char*)malloc(nrows*ncols*nchans*sizeof(unsigned char));
	unsigned char* d_debug;
	cudaMalloc(&d_debug,nchans*nrows*ncols*sizeof(unsigned char));

	// define a shortcut to the host data arrays
	unsigned char* data_l = ((unsigned char*)(im_l.data));
	unsigned char* data_r = ((unsigned char*)(im_r.data));

	//copy the host input data to the device
    cudaMemcpy(d_im_l, data_l, nchans*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im_r, data_r, nchans*nrows*ncols*sizeof(unsigned char), cudaMemcpyHostToDevice);

	// get gaussian kernel for spacial look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	// float s_weights[win_size][win_size]; 
	// for(int i=0; i<win_size; i++){
	// 	for(int j=0; j<win_size; j++){
	// 		float x = i-win_rad;
	// 		float y = j-win_rad;
	// 		float radius = sqrt(x*x+y*y);
	// 		s_weights[i][j] = std::pow(2.71828,-radius*radius/(2.*s_sigma*s_sigma));
	// 		// printf("%.6f ",s_weights[i][j]);
	// 	}
	// 	// printf("\n");
	// }

	// get gaussian kernel for color look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	// float c_weights[511]; 
	// for(int i=0; i<511; i++){
	// 	float radius = i-255;
	// 	c_weights[i] = std::pow(2.71828,-radius*radius/(2.*c_sigma*c_sigma));
	// 	// printf("%.6f ",c_weights[i]);
	// }

	// initialize the outputs (otherwise changes persist between runtimes, hard to debug):
	int tpb = 1024;
	int bpg = nrows*ncols*sizeof(unsigned char) / tpb + 1;
	printf("zeroing output images\n");
	gpu_memset<<<bpg, tpb>>>(d_out,25,nrows*ncols*sizeof(unsigned char));
	gpu_memset<<<nchans*bpg, tpb>>>(d_debug,25,nchans*nrows*ncols*sizeof(unsigned char));

	// call the kernel here!
	dim3 blocksPerGrid(1,1);
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	size_t reference_window_size = (2*win_rad+BLOCK_SIZE)*(2*win_rad+BLOCK_SIZE)*sizeof(unsigned char)*nchans;
	size_t target_window_size = (2*win_rad+ndisp+BLOCK_SIZE)*(BLOCK_SIZE+2*win_rad)*sizeof(unsigned char)*nchans;
	size_t shared_size = target_window_size+reference_window_size;
	if(shared_size > 47000){
		printf("shared size = %d\n",shared_size);
		printf("FATAL ERROR: shared_size for asw_kernel exceeds the device limit (48 kB), exiting\n");
		return 1;
	}
	// __global__ void asw_kernel(unsigned char* global_left, unsigned char* global_right, unsigned char* output, unsigned char* debug,
	//		int nrows, int ncols, int nchans, int ndisp, int win_size, int win_rad, float s_sigma, float c_sigma)
	printf("starting asw kernel\n");
	check_timer(NULL,&timer);
    asw_kernel<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_im_l, d_im_r, d_out, d_debug,
    	nrows, ncols, nchans, ndisp, win_size, win_rad, s_sigma, c_sigma);
    check_timer("asw kernel finished",&timer);

	// copy the device output data to the host
    cudaMemcpy(out, d_out, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, nrows*ncols*nchans*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // make an image and view it:
    cv::Mat im_out(nrows,ncols,CV_8UC1,out);
    cv::Mat im_debug(nrows,ncols,CV_8UC3,debug);
    cv::rectangle(im_debug,cv::Point(16*15,16*15),cv::Point(16*16,16*16),cv::Scalar(255,0,0));
    cv::rectangle(im_out,cv::Point(16*15,16*15),cv::Point(16*16,16*16),127);
    cv::imshow("window",im_out);
    cv::waitKey(0);

	// cleanup memory
	cudaFree(d_im_l);
	cudaFree(d_im_r);
	cudaFree(d_out);
	cudaFree(d_debug);
	free(out);
	free(debug);

	return 0;
}

int main(int argc, char** argv){
	// spacial and color sigmas
	int s_sigma, c_sigma;
	// number of disparities to check
	int ndisp;
	// input images
	cv::Mat im_l, im_r;

	if(argc < 6){
		printf("usage: %s <left image> <right image> <num disparities> <spacial sigma> <color sigma>",argv[0]);
		return 1;
	}else{
		im_l = cv::imread(argv[1]);
		im_r = cv::imread(argv[2]);
		ndisp = atoi(argv[3]);
		s_sigma = atoi(argv[4]);
		c_sigma = atoi(argv[5]);
	}

	return asw(im_l, im_r, ndisp, s_sigma, c_sigma);
}