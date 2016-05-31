#ifndef COSTVOLUMEFILTER_BOX_H
#define COSTVOLUMEFILTER_BOX_H

#include <opencv2/opencv.hpp>
#include "cost_volume.h"
#include "timer.h"

void costVolumeFilter_box_gpu(struct cost_volume_t& vol, int ksize);
void costVolumeFilter_box(struct cost_volume_t& cost_volume, int kernelSize);

#endif // COSTVOLUMEFILTER_BOX_H
