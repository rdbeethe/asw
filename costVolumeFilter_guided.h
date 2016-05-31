#ifndef COSTVOLUMEFILTER_GUIDED_H
#define COSTVOLUMEFILTER_GUIDED_H

#include <opencv2/opencv.hpp>
#include "cost_volume.h"
#include "timer.h"

__global__ void costVolumeFilter_guided_kernel(struct cost_volume_t vol, int* guide_global, int inter_win_padding, float* output, float sigma_s, float sigma_c, int ksize);
void costVolumeFilter_guided_gpu(struct cost_volume_t& vol, cv::Mat guide, int ksize, float eps);
void costVolumeFilter_guided(struct cost_volume_t& vol, cv::Mat guide, int ksize, float eps);

#endif // COSTVOLUMEFILTER_GUIDED_H

