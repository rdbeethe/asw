#ifndef COSTVOLUMEFILTER_JOINTBILATERAL_H
#define COSTVOLUMEFILTER_JOINTBILATERAL_H

#include <opencv2/opencv.hpp>
#include "cost_volume.h"
#include "timer.h"

__global__ void costVolumeFilter_jointBilateral_kernel(struct cost_volume_t vol, int* guide_global, int inter_win_padding, float* output, int ksize, float sigma_c, float sigma_s);
void costVolumeFilter_jointBilateral_gpu(struct cost_volume_t& cost_volume, cv::Mat guide, int ksize, float sigma_c, float sigma_s);
//void jointBilateralFilter(cv::Mat& srcim, cv::Mat& guideim, cv::Mat& dst, int kernelSize, float sigma_color, float sigma_space);

void costVolumeFilter_jointBilateral(struct cost_volume_t& cost_volume, cv::Mat guide, int kernelSize, float sigma_color, float sigma_space);

#endif // COSTVOLUMEFILTER_JOINTBILATERAL_H
