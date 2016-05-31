#ifndef COSTVOLUMEMINIMIZE_H
#define COSTVOLUMEMINIMIZE_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "timer.h"
#include "cost_volume.h"
#include "helper.h"

__global__ void costVolumeMinimize_kernel(struct cost_volume_t vol, unsigned char* output);
void costVolumeMinimize_gpu(struct cost_volume_t cost_volume, cv::Mat& outim);
void costVolumeMinimize(struct cost_volume_t cost_volume, cv::Mat& outim);

#endif // COSTVOLUMEMINIMIZE_H

