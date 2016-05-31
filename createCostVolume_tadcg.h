#ifndef CREATECOSTVOLUME_TADCG_H
#define CREATECOSTVOLUME_TADCG_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

__global__ void createCostVolume_tadcg_kernel(cv::cuda::PtrStepi ref_global, cv::cuda::PtrStepi tgt_global, cv::cuda::PtrStepf vol, int nrows, int ncols, int ndisp, float tc, float tg, float alpha);
cv::cuda::GpuMat createCostVolume_tadcg_gpu(cv::Mat leftim, cv::Mat rightim, int ndisp, float tc, float tg, float alpha);
cv::Mat createCostVolume_tadcg(cv::Mat leftim, cv::Mat rightim, int ndisp, float tc, float tg, float alpha);

#endif // CREATECOSTVOLUME_TADCG_H
