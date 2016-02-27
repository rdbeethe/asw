#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>


#define MAX_BLOCK_SIZE 32
#define MAX_WINDOW_SIZE 55
#define MAX_DISP 128

#define NCHANS 3

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define bdx blockDim.x
#define bdy blockDim.y
#define tix threadIdx.x
#define tiy threadIdx.y


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
	int gx = blockIdx.x*bdx + tix;
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
	// each thread will calculate the full asw stereo output for a single pixel
	extern __shared__ float im[]; // contains both left and right image data

	// get the size of the sub-images that we are considering
	// reference window
	int ref_width_pix = 2*win_rad+bdx;
	// target window
	int tgt_width_pix = ndisp+2*win_rad+bdx;

	float* ref = im; // left image, reference to the beginnning of im[]
	float* tgt = (float*)(&im[ ref_width_pix*NCHANS*2*bdy ]); // right image, reference to somwhere in middle of im[]

	float ref_c_factor;  // this will have to be recalculated every time
	float tgt_c_factor;  // this will have to be recalculated every time
	float s_factor; // I think I can calculate this each time for now
	// variables for keeping track of the output
	float weight;
	float weights[MAX_DISP];
	float cost;
	float costs[MAX_DISP];
	float min_cost;
	int min_cost_index;
	float ref_center_pix[3];
	float tgt_center_pix[3];
	float ref_pix[3];
	float tgt_pix[3];
	float tgt_centers[3*MAX_DISP];

	// get identity of this thread
	int bx = blockIdx.x + 5;
	int by = blockIdx.y + 1;
	int gx = bx*bdx + tix;
	int gy = by*bdy + tiy;

	// initialize weights and costs to zero
	for(int i = 0; i < ndisp; i ++){
		costs[i]=0;
		weights[i]=0;
	}

	// copy center pix for all disparities from global memory to local memory
	// since there will be a nasty access pattern
	// TODO: better boundary conditions
	for(int disp = 0; disp < ndisp; disp++){
		tgt_centers[3*disp + 0] = global_right[(gy*ncols + gx - disp)*NCHANS + 0]; 
		tgt_centers[3*disp + 1] = global_right[(gy*ncols + gx - disp)*NCHANS + 1]; 
		tgt_centers[3*disp + 2] = global_right[(gy*ncols + gx - disp)*NCHANS + 2]; 
	}

	// store ref_center_pix in register, which is constant for any given thread
	ref_center_pix[0] = global_left[gy*ncols*NCHANS + gx + 0];
	ref_center_pix[1] = global_left[gy*ncols*NCHANS + gx + 1];
	ref_center_pix[2] = global_left[gy*ncols*NCHANS + gx + 2];

	// first copy from global memory to the first bdy rows of shared memory
	// in the ref image... :
	for(int x_block = 0; x_block < (2*win_rad+bdx)/bdx + 1; x_block++){
		// check if x is greater than the width of the reference image
		if(x_block*bdx + tix > ref_width_pix*NCHANS){
			ref[ref_width_pix*NCHANS*tiy + x_block*bdx + tix] = global_left[((gy-win_rad)*ncols+gx-win_rad+x_block*bdx)*NCHANS];
		}
	}
	__syncthreads();
	// ... and in the tgt image:
	for(int x_block = 0; x_block < (2*win_rad+bdx+ndisp)/bdx + 1; x_block++){
		// check if x is greater than the width of the tgt image
		if(x_block*bdx + tix > tgt_width_pix*NCHANS){
			tgt[tgt_width_pix*NCHANS*tiy + x_block*bdx + tix] = global_right[((gy-win_rad)*ncols+gx-win_rad-ndisp+x_block*bdx)*NCHANS];
		}
	}
	__syncthreads();

	///////// MAIN LOOP:
	// for every block copy from global memory
	for(int y_block = 0; y_block < (2*win_rad+bdy)/(2*bdy)+1; y_block ++){
		// check to make sure we're in a row that needs to be copied
		if(y_block*(bdy+1) + tiy < 2*win_rad + bdy){
			// copy from global memory to the last bdy rows of shared memory 
			// in the ref image... :
			for(int x_block = 0; x_block < (2*win_rad+bdx)/bdx + 1; x_block++){
				ref[ref_width_pix*NCHANS*tiy + x_block*bdx + tix] = global_left[((gy-win_rad+y_block*(bdy+1))*ncols+gx-win_rad+x_block*bdx)*NCHANS];
			}
			// ... then in the tgt image:
			for(int x_block = 0; x_block < (2*win_rad+bdx+ndisp)/bdx + 1; x_block++){
				tgt[tgt_width_pix*NCHANS*tiy + x_block*bdx + tix] = global_right[((gy-win_rad+y_block*(bdy+1))*ncols+gx-win_rad-ndisp+x_block*bdx)*NCHANS];
			}
		}
		__syncthreads();
		// now for up to bdy rows of shared memory:
		int shared_y = 0;
		while(true){
			// check if this is past the last valid row
			if(y_block*bdy + shared_y == win_size){
				// TODO: really this should skip the final copy step, just after this loop
				break;
			}else if(shared_y > bdy && y_block*bdy + shared_y != win_size-1){ // otherwise, unless it is the very last row...
				// then completing bdy calculations means its time to move on
				break;
			}
			// for each value of ndisp	
			for(int disp = 0; disp < ndisp; disp++){

                // TODO: this seems like it's not the right numbers output
                // if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0){
                //     ((int*)debug)[(2+y_block)*ndisp+disp] = shared_y;
                // }
                // __syncthreads();

				// store tgt_center_pix in register memory, which is constant for each disp
				tgt_center_pix[0] = tgt_centers[3*disp + 0];
				tgt_center_pix[1] = tgt_centers[3*disp + 1];
				tgt_center_pix[2] = tgt_centers[3*disp + 2];
				// reset weight and cost
				weight = 0;
				cost = 0;
				// in each column of the window:
				for(int win_x = 0; win_x < win_size; win_x++){
					// get the s_factor for this particular window location
					s_factor = __expf(-((float)(win_x-win_rad)*(win_x-win_rad) + (shared_y+y_block*bdy-win_rad)*(shared_y+y_block*bdy-win_rad))/(2.*s_sigma*s_sigma));
					// store tgt and ref pixels in register memory
					ref_pix[0] = ref[(shared_y+tiy)*ref_width_pix*NCHANS + (win_x+tix)*NCHANS + 0];
					ref_pix[1] = ref[(shared_y+tiy)*ref_width_pix*NCHANS + (win_x+tix)*NCHANS + 1];
					ref_pix[2] = ref[(shared_y+tiy)*ref_width_pix*NCHANS + (win_x+tix)*NCHANS + 2];
					tgt_pix[0] = tgt[(shared_y+tiy)*tgt_width_pix*NCHANS + (ndisp+win_x+tix-disp)*NCHANS + 0];
					tgt_pix[1] = tgt[(shared_y+tiy)*tgt_width_pix*NCHANS + (ndisp+win_x+tix-disp)*NCHANS + 1];
					tgt_pix[2] = tgt[(shared_y+tiy)*tgt_width_pix*NCHANS + (ndisp+win_x+tix-disp)*NCHANS + 2];
					// get the center-to-pixel and overall color differences (organized together for ILP)
					float ref_c2p_diff = abs(ref_center_pix[0] - ref_pix[0]);
					float tgt_c2p_diff = abs(tgt_center_pix[0] - ref_pix[0]);
					float ref2tgt_diff = abs(ref_pix[0] - tgt_pix[0]);
					// include additional channels
					ref_c2p_diff += abs(ref_center_pix[1] - ref_pix[1]);
					tgt_c2p_diff += abs(tgt_center_pix[1] - ref_pix[1]);
					ref2tgt_diff+= abs(ref_pix[1] - tgt_pix[1]);
					ref_c2p_diff += abs(ref_center_pix[2] - ref_pix[2]);
					tgt_c2p_diff += abs(tgt_center_pix[2] - ref_pix[2]);
					ref2tgt_diff+= abs(ref_pix[2] - tgt_pix[2]);
					// get the c_factors
					ref_c_factor = __expf(-ref_c2p_diff*ref_c2p_diff/(2.*c_sigma*c_sigma));
					tgt_c_factor = __expf(-tgt_c2p_diff*tgt_c2p_diff/(2.*c_sigma*c_sigma));
					// add in the cost
					cost += s_factor*ref_c_factor*tgt_c_factor*ref2tgt_diff;
					// add in the weight
					weight += s_factor*ref_c_factor*tgt_c_factor;
				}
				// before moving to a new disp, add the weights and costs into a local memory array
				costs[disp] += cost;
				weights[disp] += weight;
			}
			// move on to the next row of shared_y
			shared_y ++;
		}
		// copy the bottom half of the shared memory array to the top half
		// first in the ref image... :
		for(int x_block = 0; x_block < (2*win_rad+bdx)/bdx + 1; x_block++){
			// check if x is greater than the width of the reference image
			if(x_block*bdx + tix > ref_width_pix*NCHANS){
				ref[ref_width_pix*NCHANS*tiy + x_block*bdx + tix] = ref[ref_width_pix*NCHANS*(tiy+bdy) + x_block*bdx + tix];
			}
		}
		__syncthreads();
		// ... and in the tgt image:
		for(int x_block = 0; x_block < (2*win_rad+bdx+ndisp)/bdx + 1; x_block++){
			// check if x is greater than the width of the tgt image
			if(x_block*bdx + tix > tgt_width_pix*NCHANS){
				tgt[tgt_width_pix*NCHANS*tiy + x_block*bdx + tix] = tgt[tgt_width_pix*NCHANS*(tiy+bdy) + x_block*bdx + tix];
			}
		}
		__syncthreads();
	}


	// now go through and find the lowest normalized cost
	min_cost_index = 0;
	min_cost = costs[0]/weights[0];
	for(int disp = 1; disp < ndisp; disp++){
		cost = costs[disp]/weights[disp];	
		if(cost < min_cost){
			min_cost = cost;
			min_cost_index = disp;
		}
        // if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0){
        //     ((float*)debug)[disp] = costs[disp];
        //     ((float*)debug)[ndisp+disp] = bdy;
        //     // ((float*)debug)[disp+ndisp] = weights[disp];
        // }
		// __syncthreads();
	}

	// set the output to min_cost_index
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

	// initialize the outputs (otherwise changes persist between runtimes, hard to debug):
	int tpb = 1024;
	int bpg = nrows*ncols*sizeof(unsigned char) / tpb + 1;
	printf("zeroing output images\n");
	gpu_memset<<<bpg, tpb>>>(d_out,25,nrows*ncols*sizeof(unsigned char));
	gpu_perror("memset1");
	gpu_memset<<<nchans*bpg, tpb>>>(d_debug,25,nchans*nrows*ncols*sizeof(unsigned char));
	gpu_perror("memset2");

	// check some values before calling the asw_kernel
	size_t reference_window_size = (2*win_rad+BLOCK_SIZE_X)*(2*BLOCK_SIZE_Y)*sizeof(float)*nchans;
	size_t target_window_size = (2*win_rad+ndisp+BLOCK_SIZE_X)*(2*BLOCK_SIZE_Y)*sizeof(float)*nchans;
	size_t shared_size = target_window_size+reference_window_size;
	printf("win_size %d win_rad %d ndisp %d shared size = %d\n",win_size,win_rad,ndisp,shared_size);
	if(shared_size > 47000){
		printf("FATAL ERROR: shared_size for asw_kernel exceeds the device limit (48 kB), exiting\n");
		return 1;
	}

	{
		// call the asw_kernel
		dim3 blocksPerGrid(21*16/BLOCK_SIZE_X,22*16/BLOCK_SIZE_Y);
		// dim3 blocksPerGrid(1,1);
		dim3 threadsPerBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
		// __global__ void asw_kernel(unsigned char* global_left, unsigned char* global_right, unsigned char* output, unsigned char* debug,
		//		int nrows, int ncols, int nchans, int ndisp, int win_size, int win_rad, float s_sigma, float c_sigma)
		printf("starting asw kernel\n");
		check_timer(NULL,&timer);
	    asw_kernel<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_im_l, d_im_r, d_out, d_debug,
	    	nrows, ncols, nchans, ndisp, win_size, win_rad, s_sigma, c_sigma);
	    cudaDeviceSynchronize();
	    check_timer("asw kernel finished",&timer);
		gpu_perror("asw_kernel");
	}

	// copy the device output data to the host
	check_timer(NULL,&timer);
    cudaMemcpy(out, d_out, nrows*ncols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug, d_debug, nrows*ncols*nchans*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    check_timer("copying complete",&timer);
    printf("%d\n",debug[0]);

    // make an image and view it:
    cv::Mat im_out(nrows,ncols,CV_8UC1,out);
    cv::Mat im_debug(nrows,ncols,CV_8UC3,debug);
    // cv::rectangle(im_debug,cv::Point(16*15,16*15),cv::Point(16*16,16*16),cv::Scalar(255,0,0));
    // cv::rectangle(im_out,cv::Point(16*15,16*15),cv::Point(16*16,16*16),127);
    // cv::imshow("window",im_debug);
    // cv::waitKey(0);
    // cv::imshow("window",im_out);
    // cv::waitKey(0);
    cv::imwrite("out.png",im_out);
    // for(int i=0; i<ndisp; i++){
    //     printf("index:cost:weight for disp %d:%f:%f\n", i, ((float*)debug)[i],((float*)debug)[ndisp+i]);
    // }

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
