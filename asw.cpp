#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>

using namespace std;
using namespace cv;

#define NUMTHREADS 4

int argmin_double(double* data, int len){
	double min = data[0];
	int idx = 0;
	// double max = data[0];
	// int maxidx = 0;
	for(int i=1; i<len; i++){
		if(data[i] < min){
			min = data[i];
			idx = i;
		}
		// if(data[i] > max){
		// 	max = data[i];
		// 	maxidx = i;
		// }
	}
	return idx;
}

struct asw_arg_t {
	int win_size;
	int win_rad;
	int ndisp;
	float s_sigma;
	float i_sigma;
	int nrows;
	int ncols;
	int nchans;
	unsigned char* out;
	unsigned char* l;
	unsigned char* r;
	unsigned char* dxl;
	unsigned char* dxr;
	int xmin;
	int ymin;
	int xmax;
	int ymax;
};

void* p_thread_asw(void* arg_ptr){
	// dereference the argument, and make local variables out of all of the parts
	struct asw_arg_t* arg = (struct asw_arg_t*)arg_ptr;
	int win_size = arg->win_size;
	int win_rad = arg->win_rad;
	int ndisp = arg->ndisp;
	float s_sigma = arg->s_sigma;
	float i_sigma = arg->i_sigma;
	int nrows = arg->nrows;
	int ncols = arg->ncols;
	int nchans = arg->nchans;
	unsigned char* out = arg->out;
	unsigned char* l = arg->l;
	unsigned char* r = arg->r;
	unsigned char* dxl = arg->dxl;
	unsigned char* dxr = arg->dxr;
	int xmin = arg->xmin;
	int ymin = arg->ymin;
	int xmax = arg->xmax;
	int ymax = arg->ymax;

	// some local copies of LUT's (so as to avoid read conflicts, if that's a thing in CPU-land)

	// get gaussian kernel for spacial look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	double s_weights[win_size][win_size]; 
	for(int i=0; i<win_size; i++){
		for(int j=0; j<win_size; j++){
			double x = i-win_rad;
			double y = j-win_rad;
			double radius = sqrt(x*x+y*y);
			s_weights[i][j] = std::pow(2.71828,-radius*radius/(2.*s_sigma*s_sigma));
			// printf("%.6f ",s_weights[i][j]);
		}
		// printf("\n");
	}

	// get gaussian kernel for intensity look-up table:
	// equation from cv::getGaussianKernel(), but without normalization
	double i_weights[511]; 
	for(int i=0; i<511; i++){
		double radius = i-255;
		i_weights[i] = std::pow(2.71828,-radius*radius/(2.*i_sigma*i_sigma));
		// printf("%.6f ",i_weights[i]);
	}

	// TAD C+G (truncated abs. diff. of color and gradient) values
	int Tc = i_sigma;
	int Tg = 20;
	// color weight
	double alpha = 1;

	// uniqeness requirement
	double min_match = 1.0;

	// prepare for the list of costs at each disparity
	double* costs = (double*)malloc(ndisp*sizeof(double));

	// for debugging, double-check how image was subdividied
	// printf("x : %d,%d    y : %d,%d\n",xmin,xmax,ymin,ymax);

	// do asw
	// first two layers are to touch every pixel:
	for(int row = ymin; row < ymax; row++){
		for(int col = xmin; col < xmax; col++){
			// costs represnts the matching costs for this scanline:
			// this layer is to scan along the search region:
			for(int disp = 0; disp < min(ndisp,col); disp++){
				// define floats for tracking this pixel's matching cost:
				double cost = 0;
				double normalizing_factor = 0;
				// get local pointers for l and r center pixels
				unsigned char* center_l = &l[(row*ncols + col)*nchans];
				unsigned char* center_r = &r[(row*ncols + col - disp)*nchans];
				unsigned char* center_dxl = &dxl[(row*ncols + col)*nchans];
				unsigned char* center_dxr = &dxr[(row*ncols + col - disp)*nchans];
				// the next two layers are to touch all neighborhood pixels:
				for(int j = max(0,row-win_rad); j < min(nrows,row+win_rad+1); j++){
					for(int i = max(disp,col-win_rad); i < min(ncols,col+win_rad+1); i++){
						// find the local variation coordinates
						int x = i - col;
						int y = j - row;
						// get a pointer to the variation pixel
						unsigned char* pixel_l = &l[( j * ncols + i ) * nchans];
						unsigned char* pixel_r = &r[( j * ncols + i - disp ) * nchans];
						unsigned char* pixel_dxl = &dxl[( j * ncols + i ) * nchans];
						unsigned char* pixel_dxr = &dxr[( j * ncols + i - disp ) * nchans];
						// initialize the left and right weight with spacial sigma
						double weight_l = s_weights[x+win_rad][y+win_rad];
						double weight_r = weight_l;
						// double weight_r = 1;
						// also, initialize the sum of abs diff between windows
						int sad = 0;
						int dxsad = 0;
						double sad_total;
						// this loop is to touch each color channel
						for(int chan = 0; chan < nchans; chan++){
							//get the intensity abs difference for left and right
							int diff_l = abs(((int)pixel_l[chan]) - ((int)center_l[chan]));
							int diff_r = abs(((int)pixel_r[chan]) - ((int)center_l[chan]));
							// int diff_r = abs(((int)pixel_r[chan]) - ((int)center_r[chan]));
							// add the abs difference between the two windows
							sad += abs(((int)pixel_l[chan]) - ((int)pixel_r[chan]));
							// multiply in the weight from this channel
							weight_l *= i_weights[diff_l+255];
							weight_r *= i_weights[diff_r+255];
						}
						// get abs diff of gradient
						dxsad = abs(((int)pixel_dxl[0]) - ((int)center_dxr[0]));
						// truncate values before adding
						if(sad > Tc){
							sad = Tc;
						}
						if(dxsad > Tg){
							dxsad = Tg;
						}
						// combine values
						sad_total = alpha*sad + (1.-alpha)*dxsad;
						// we're done with this variation pixel:
						// add the weight times the difference to the cost
						cost += sad_total*weight_l*weight_r;
						normalizing_factor += weight_l*weight_r;
						// printf("normalizing_factor:%f\n",normalizing_factor);
					}
				}
				// add the cost for this window the list of costs
				costs[disp] = cost/normalizing_factor;
			}
			// find best value
			int best_idx = argmin_double(costs,ndisp);
			double best_val = costs[best_idx];
			// find next best val
			costs[best_idx] = 1e10;
			int next_best_idx = argmin_double(costs,ndisp);
			double next_best_val = costs[next_best_idx];
			// set output image value, to black if it's not a good match
			if(next_best_val/best_val > min_match){
				out[row*ncols + col] = ((char)argmin_double(costs,ndisp));
			}else{
				out[row*ncols + col] = 0;
			}
		}
	}

	// free the costs array
	free(costs);

	pthread_exit(NULL);
}


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

char myshow(Mat im){
	double minval,maxval;
	minMaxLoc(im,&minval, &maxval);
	Mat im2 = im - minval;
	im2.convertTo(im2, CV_8U, 255./(maxval - minval));
	imshow("window",im2);
	return (char)waitKey(0);
}

int asw(Mat l_im, Mat r_im, int ndisp, float s_sigma, float i_sigma){
	// declare timer
	struct timespec timer;


	// window size and win_rad
	int win_size = 3*s_sigma-1;
	int win_rad = (win_size - 1)/2;

	// check that images are matching dimensions
	if(l_im.rows != r_im.rows){
		printf("Error: l_im and r_im do not have matching row count\n");
		return 1;
	}
	if(l_im.cols != r_im.cols){
		printf("Error: l_im and r_im do not have matching col count\n");
		return 1;
	}
	if(l_im.channels() != r_im.channels()){
		printf("Error: l_im and r_im do not have matching channel count\n");
		return 1;
	}

	// calculate image derivatives
	Mat dxl_im, dxr_im;
	// void Sobel(src, dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )¶
	Sobel(l_im,dxl_im, CV_16S, 1, 0, CV_SCHARR);
	Sobel(r_im,dxr_im, CV_16S, 1, 0, CV_SCHARR);
	double minval,maxval;
	minMaxLoc(dxl_im,&minval, &maxval);
	printf("%f,%f",minval,maxval);

	// set easy-access variables for number of rows, cols, and chans
	int nrows = l_im.rows;
	int ncols = l_im.cols;
	int nchans = l_im.channels();
	// initialize the output data matrix
	unsigned char* out = (unsigned char*)malloc(nrows*ncols*sizeof(unsigned char));

	// define a shortcut to the data arrays
	unsigned char* l = ((unsigned char*)(l_im.data));
	unsigned char* r = ((unsigned char*)(r_im.data));
	unsigned char* dxl = ((unsigned char*)(dxl_im.data));
	unsigned char* dxr = ((unsigned char*)(dxr_im.data));

	// full values
	// int xmin = 0;
	// int xmax = nrows;
	// int ymin = 0;
	// int ymax = ncols;
	// shorter values
	int xmin = 64;
	int xmax = 200;
	int ymin = 100;
	int ymax = 275;

	// prepare for pthread calls
	struct asw_arg_t asw_args[NUMTHREADS];
	for(int i = 0; i < NUMTHREADS; i++){
		asw_args[i].win_size = win_size;
		asw_args[i].win_rad = win_rad;
		asw_args[i].ndisp = ndisp;
		asw_args[i].s_sigma = s_sigma;
		asw_args[i].i_sigma = i_sigma;
		asw_args[i].nrows = nrows;
		asw_args[i].ncols = ncols;
		asw_args[i].nchans = nchans;
		asw_args[i].out = out;
		asw_args[i].l = l;
		asw_args[i].r = r;
		asw_args[i].dxl = dxl;
		asw_args[i].dxr = dxr;
		// partition the image into horizontal sub-images
		asw_args[i].xmin = xmin;
		asw_args[i].ymin = ymin + 1.*i*(ymax-ymin)/NUMTHREADS;
		asw_args[i].xmax = xmax;
		asw_args[i].ymax = ymin + 1.*(i+1)*(ymax-ymin)/NUMTHREADS;
	}

	pthread_t threads[NUMTHREADS];

	// start timer
	check_timer(NULL,&timer);

	for(int i = 0; i < NUMTHREADS; i++){
		pthread_create(&threads[i],NULL,p_thread_asw,&asw_args[i]);
	}
	for(int i = 0; i < NUMTHREADS; i++){
		pthread_join(threads[i], NULL);
	}

	// print timer data
	check_timer("pthread CPU asw: ",&timer);

	//show the output matrix
	Mat outmat(nrows,ncols,CV_8UC1,out);
	myshow(outmat);
	imwrite("out_cpu_asw.png",outmat);

	free(out);

	return 0;
}

int main(int argc, char** argv){
	// spacial and intensity sigmas
	int s_sigma, i_sigma;
	// number of disparities to check
	int ndisp;
	// input images
	Mat l_im, r_im;

	if(argc < 6){
		printf("usage: %s <left image> <right image> <num disparities> <spacial sigma> <color sigma>\n\n",argv[0]);
		printf("... for now, using defaults (l.png r.png 64 5 50)\n");
		l_im = imread("l.png");
		r_im = imread("r.png");
		ndisp = 64;
		s_sigma = 10;
		i_sigma = 100;
	}else{
		l_im = imread(argv[1]);
		r_im = imread(argv[2]);
		ndisp = atoi(argv[3]);
		s_sigma = atoi(argv[4]);
		i_sigma = atoi(argv[5]);
	}

	return asw(l_im, r_im, ndisp, s_sigma, i_sigma);
}