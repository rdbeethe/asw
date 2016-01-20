#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>


#define MAX_BLOCK_SIZE 32
#define MAX_WINDOW_SIZE 55
#define MAX_DISP 1000

#define NCHANS 3

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

// teeny little helper function
void gpu_perror(char* input){
	printf("%s: %s\n", input, cudaGetErrorString(cudaGetLastError()));
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
	extern __shared__ unsigned char ref[]; // contains both left and right image data

	// get the size of the sub-images that we are considering
	// reference window
	int ref_width_bytes = (2*win_rad+blockDim.x)*NCHANS*sizeof(unsigned char);
	// int ref_rows = (2*win_rad+blockDim.y);
	// target window
	int tgt_width_bytes = (ndisp+2*win_rad+blockDim.x)*NCHANS*sizeof(unsigned char);
	// int tgt_rows = (2*win_rad+blockDim.y);

	unsigned char* tgt = (unsigned char*)(&ref[ ref_width_bytes*(2*win_rad+blockDim.y) ]); // tgt image, reference to somwhere of shared allocated memory

	float ref_c_factor;
	float tgt_c_factor;
	float s_factor;
	float ref_c2p_diff;
	float tgt_c2p_diff;
	float ref2tgt_diff;
	// variables for keeping track of the output
	float weight;
	float cost;
	float min_cost;
	unsigned char min_cost_index;
	unsigned char ref_center_pix[3];
	unsigned char tgt_center_pix[3];
	unsigned char ref_pix[3];
	unsigned char tgt_pix[3];

	int disp;
	int win_x;
	int win_y;
	int dx;
	int tgt_x;

	// get identity of this thread (changing these to #define's)

	#define tx (threadIdx.x)
	#define ty (threadIdx.y)
	#define bx (blockIdx.x + 5)
	#define by (blockIdx.y + 1)
	#define gx (bx*blockDim.x + tx)
	#define gy (by*blockDim.y + ty)

	// setup LUTs // nevermind... right now there are none

	// copy relevant subimages to shared memory
	// TODO: additional boundary checks on this data
	// TODO: better division technique
	// TODO: investigate where syncthreads() needs to be called for best performance
	// we can copy the 24-bit image over 32 bits at a time
	// except then I don't know how to deal with the edge case
	// so let's just do one character at a time
	// starting with reference image: (4 deleted register variables)	
	// int xblocks = (ref_width_bytes / blockDim.x + 1);
	// int yblocks = ((2*win_rad+blockDim.y) / blockDim.y + 1);
	// int xstart = ((bx*blockDim.x - win_rad)*NCHANS);
	// int ystart = (gy - win_rad);
	// 29 variables here
	for(win_x = 0; win_x < (ref_width_bytes / blockDim.x + 1); win_x++){
		// int x_idx = (win_x*blockDim.x + tx);
		// int g_x_idx = (((bx*blockDim.x - win_rad)*NCHANS) + win_x*blockDim.x + tx);
		if((win_x*blockDim.x + tx) < ref_width_bytes){
			for(win_y = 0; win_y < ((2*win_rad+blockDim.y) / blockDim.y + 1); win_y++){
				// int y_idx = (win_y*blockDim.y + ty);
				// int g_y_idx = ((gy - win_rad) + win_y*blockDim.y);
				if((win_y*blockDim.y + ty) < (2*win_rad+blockDim.y)){
					// copy bytes (not pixels) from global_left into reference image
					ref[(win_y*blockDim.y + ty)*ref_width_bytes + (win_x*blockDim.x + tx)] = global_left[((gy - win_rad) + win_y*blockDim.y)*ncols*NCHANS + (((bx*blockDim.x - win_rad)*NCHANS) + win_x*blockDim.x + tx)];
					// copy into the debug image (only made to work with a single block of threads)
					// debug[((gy - win_rad) + win_y*blockDim.y)*ncols*NCHANS + (((bx*blockDim.x - win_rad)*NCHANS) + win_x*blockDim.x + tx)]  = ref[(win_y*blockDim.y + ty)*ref_width_bytes + (win_x*blockDim.x + tx)];
				}
			}
		}
	}
	// then to the target image: (4 deleted register variables)
	// xblocks = (tgt_width_bytes / blockDim.x + 1);
	// yblocks = ((2*win_rad+blockDim.y) / blockDim.y + 1);
	// xstart = ((bx*blockDim.x - win_rad - ndisp)*NCHANS);
	// ystart = (gy - win_rad);
	for(win_x = 0; win_x < (tgt_width_bytes / blockDim.x + 1); win_x++){
		// int x_idx = (win_x*blockDim.x + tx);
		// int g_x_idx = (((bx*blockDim.x - win_rad - ndisp)*NCHANS) + win_x*blockDim.x + tx);
		if((win_x*blockDim.x + tx) < tgt_width_bytes){
			for(win_y = 0; win_y < ((2*win_rad+blockDim.y) / blockDim.y + 1); win_y++){
				// int y_idx = (win_y*blockDim.y + ty);
				// int g_y_idx = ((gy - win_rad) + win_y*blockDim.y);
				if((win_y*blockDim.y + ty) < (2*win_rad+blockDim.y)){
					// copy bytes (not pixels) from global_left into reference image
					tgt[(win_y*blockDim.y + ty)*tgt_width_bytes + (win_x*blockDim.x + tx)] = global_right[((gy - win_rad) + win_y*blockDim.y)*ncols*NCHANS + (((bx*blockDim.x - win_rad - ndisp)*NCHANS) + win_x*blockDim.x + tx)];
					// copy into the debug image (only made to work with a single block of threads)
					// debug[((gy - win_rad) + win_y*blockDim.y)*ncols*NCHANS + (((bx*blockDim.x - win_rad - ndisp)*NCHANS) + win_x*blockDim.x + tx)]  = tgt[(win_y*blockDim.y + ty)*tgt_width_bytes + (win_x*blockDim.x + tx)];
				}
			}
		}
	}

	__syncthreads();

	// get a pointer to the ref_center_pix, which is constant for any given thread
	ref_center_pix[0] = ref[(win_rad + ty)*ref_width_bytes + (win_rad + tx)*NCHANS + 0];
	ref_center_pix[1] = ref[(win_rad + ty)*ref_width_bytes + (win_rad + tx)*NCHANS + 1];
	ref_center_pix[2] = ref[(win_rad + ty)*ref_width_bytes + (win_rad + tx)*NCHANS + 2];
	// initialize min_cost to some arbitrarily large value
	min_cost = 1e12;
	// initialize min_cost_index to 0
	min_cost_index = 0;

	// for each value of ndisp	
	for(disp = 0; disp < ndisp; disp++){
		// get a pointer to the tgt_center_pix, which is constant for each disp
		tgt_center_pix[0] = tgt[(win_rad + ty)*tgt_width_bytes + (ndisp + win_rad + tx - disp)*NCHANS + 0];
		tgt_center_pix[1] = tgt[(win_rad + ty)*tgt_width_bytes + (ndisp + win_rad + tx - disp)*NCHANS + 1];
		tgt_center_pix[2] = tgt[(win_rad + ty)*tgt_width_bytes + (ndisp + win_rad + tx - disp)*NCHANS + 2];
		// reset weight and cost
		weight = 0;
		cost = 0;
		// in each row in the window:
		for(win_x = 0; win_x < win_size; win_x++){
			// locate the pixel in the ref image (deleted this var)
			dx = win_x + tx;
			// locate the pixel in the tgt image (deleted this var)
			tgt_x = ndisp + win_x + tx - disp;
			// find the window-center to pixel x-distance (deleted this var)
			// int dx = win_x - win_rad;
			// in each column of the window:
			for(win_y = 0; win_y < win_size; win_y++){
				// locate the pixel in the ref image (deleted this var)
				// int ref_y = win_y + ty;
				// find the window-center to pixel y-distance (deleted this var)
				// int dy = win_y - win_rad;
				// get the radius^2 value (deleted this var)
				// float radius_2 = (win_x-win_rad)*(win_x-win_rad) + (win_y-win_rad)*(win_y-win_rad);
				// get the s_factor for this particular window location
				s_factor = __expf(-((win_x-win_rad)*(win_x-win_rad) + (win_y-win_rad)*(win_y-win_rad))/(2.*s_sigma*s_sigma));
				// store tgt and ref pixels in register memory
				ref_pix[0] = ref[(win_y+ty)*ref_width_bytes + (dx)*NCHANS + 0];
				ref_pix[1] = ref[(win_y+ty)*ref_width_bytes + (dx)*NCHANS + 1];
				ref_pix[2] = ref[(win_y+ty)*ref_width_bytes + (dx)*NCHANS + 2];
				tgt_pix[0] = tgt[(win_y+ty)*tgt_width_bytes + (tgt_x)*NCHANS + 0];
				tgt_pix[1] = tgt[(win_y+ty)*tgt_width_bytes + (tgt_x)*NCHANS + 1];
				tgt_pix[2] = tgt[(win_y+ty)*tgt_width_bytes + (tgt_x)*NCHANS + 2];
				// get the center-to-pixel and overall color differences (organized together for IDP)
				ref_c2p_diff = abs(ref_center_pix[0] - ref_pix[0]);
				tgt_c2p_diff = abs(tgt_center_pix[0] - ref_pix[0]);
				ref2tgt_diff = abs(ref_pix[0] - tgt_pix[0]);
				ref_c2p_diff += abs(ref_center_pix[1] - ref_pix[1]);
				tgt_c2p_diff += abs(tgt_center_pix[1] - ref_pix[1]);
				ref2tgt_diff+= abs(ref_pix[1] - tgt_pix[1]);
				ref_c2p_diff += abs(ref_center_pix[2] - ref_pix[2]);
				tgt_c2p_diff += abs(tgt_center_pix[2] - ref_pix[2]);
				ref2tgt_diff+= abs(ref_pix[2] - tgt_pix[2]);
				// get the c_factors
				ref_c_factor = __expf(-ref_c2p_diff*ref_c2p_diff/(2.*c_sigma*c_sigma));
				tgt_c_factor = __expf(-tgt_c2p_diff*tgt_c2p_diff/(2.*c_sigma*c_sigma));
				// calulate the pix_weight (this variable has been done away with to increase ILP)
				// pix_weight = s_factor*ref_c_factor*tgt_c_factor;
				// add in the cost
				cost += s_factor*ref_c_factor*tgt_c_factor*ref2tgt_diff;
				// add in the weight
				weight += s_factor*ref_c_factor*tgt_c_factor;
			}
		}
		// now that the window is done, compare this cost (after normalizing) to min_cost
		if( min_cost > cost / weight){
			min_cost = cost / weight;
			min_cost_index = disp;
		}
		__syncthreads();
	}

	// set the output to the index of min_cost
	output[gy*ncols + gx] = min_cost_index;
}

int asw(cv::Mat im_l, cv::Mat im_r, int ndisp, int s_sigma, int c_sigma){
	// window size and win_rad
	int win_size = 3*s_sigma;
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
	gpu_perror("memset1");
	gpu_memset<<<nchans*bpg, tpb>>>(d_debug,25,nchans*nrows*ncols*sizeof(unsigned char));
	gpu_perror("memset2");

	// check some values before calling the asw_kernel
	size_t reference_window_size = (2*win_rad+BLOCK_SIZE)*(2*win_rad+BLOCK_SIZE)*sizeof(unsigned char)*nchans;
	size_t target_window_size = (2*win_rad+ndisp+BLOCK_SIZE)*(BLOCK_SIZE+2*win_rad)*sizeof(unsigned char)*nchans;
	size_t shared_size = target_window_size+reference_window_size;
	printf("win_size %d win_rad %d ndisp %d shared size = %d\n",win_size,win_rad,ndisp,shared_size);
	if(shared_size > 47000){
		printf("FATAL ERROR: shared_size for asw_kernel exceeds the device limit (48 kB), exiting\n");
		return 1;
	}

	// call the asw_kernel
	dim3 blocksPerGrid(22,21);
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	// __global__ void asw_kernel(unsigned char* global_left, unsigned char* global_right, unsigned char* output, unsigned char* debug,
	//		int nrows, int ncols, int nchans, int ndisp, int win_size, int win_rad, float s_sigma, float c_sigma)
	printf("starting asw kernel\n");
	check_timer(NULL,&timer);
    asw_kernel<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_im_l, d_im_r, d_out, d_debug,
    	nrows, ncols, nchans, ndisp, win_size, win_rad, s_sigma, c_sigma);
    cudaDeviceSynchronize();
    check_timer("asw kernel finished",&timer);
	gpu_perror("asw_kernel");

	// copy the device output data to the host
	check_timer(NULL,&timer);
    cudaMemcpy(out, d_out, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, nrows*ncols*nchans*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    check_timer("copying complete",&timer);

    // make an image and view it:
    cv::Mat im_out(nrows,ncols,CV_8UC1,out);
    cv::Mat im_debug(nrows,ncols,CV_8UC3,debug);
    // cv::rectangle(im_debug,cv::Point(16*15,16*15),cv::Point(16*16,16*16),cv::Scalar(255,0,0));
    // cv::rectangle(im_out,cv::Point(16*15,16*15),cv::Point(16*16,16*16),127);
    // cv::imshow("window",im_debug);
    // cv::waitKey(0);
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