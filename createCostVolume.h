#ifndef CREATECOSTVOLUME_H
#define CREATECOSTVOLUME_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

__global__ void createCostVolume_kernel(cv::cuda::PtrStepi ref_global, cv::cuda::PtrStepi tgt_global, cv::cuda::PtrStepf volume, int nrows, int ncols, int ndisp);
cv::cuda::GpuMat createCostVolume_gpu(cv::Mat leftim, cv::Mat rightim, int ndisp);
cv::Mat createCostVolume(cv::Mat leftim, cv::Mat rightim,int ndisp);
struct cost_volume_t cost_volume_from_gpumat(cv::cuda::GpuMat gpumat, int ndisp);
struct cost_volume_t cost_volume_from_mat(cv::Mat mat, int ndisp);

#endif // CREATECOSTVOLUME_H
