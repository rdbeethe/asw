#ifndef CREATECOSTVOLUME_H
#define CREATECOSTVOLUME_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

__global__ void createCostVolume_kernel(int* ref_global, int* tgt_global, struct cost_volume_t vol, int* debug);
struct cost_volume_t createCostVolume_gpu(cv::Mat leftim, cv::Mat rightim, int ndisp);
struct cost_volume_t createCostVolume(cv::Mat leftim, cv::Mat rightim,int ndisp);

#endif // CREATECOSTVOLUME_H
