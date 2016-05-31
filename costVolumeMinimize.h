#ifndef COSTVOLUMEMINIMIZE_H
#define COSTVOLUMEMINIMIZE_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "timer.h"
#include "cost_volume.h"
#include "helper.h"

__global__ void costVolumeMinimize_kernel(cv::cuda::PtrStepf vol, cv::cuda::PtrStep<unsigned char> output, int nrows, int ndisp, int ncols);
//__global__ void costVolumeMinimize_kernel(struct cost_volume_t vol, unsigned char* output);
void costVolumeMinimize_gpu(struct cost_volume_t cost_volume, cv::Mat& outim);
void costVolumeMinimize_gpu(cv::cuda::GpuMat cost_volume, cv::Mat& outim, int ndisp);
void costVolumeMinimize(struct cost_volume_t cost_volume, cv::Mat& outim);

#endif // COSTVOLUMEMINIMIZE_H

