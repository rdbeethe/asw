# Adaptive Support Weight (ASW) correspondence matching

This git project is an open-source CUDA implementation of the algorithm described in "Adaptive support-weight approach for correspondence" by Kuk-Jin Yoon and In So Kweon, which is the basis for the most effective local reasoning stereo vision algorithms produced today.  The latest effort is based on the articles "Fast cost-volume filtering for visual correspondence and beyond" by Rhemann et al and "Secrets of adaptive support weight for stereo vision for local stereo matching" by Hosni et al.


## Goals and Motivations

To my knowledge, there is no other open-source implementation of ASW, although it was developed in 2005.  Freely-available computer vision libraries (such as OpenCV or Nvidia VisionWorks) do not offer ASW, either in CPU form or a GPU accelerated form.  The goal of this project is to implement a GPU-accelerated ASW algorithm which can ultimately be contributed to the OpenCV library.  A Free and Open Source Software (FOSS) implementation of ASW would empower those in industry with a more powerful stereo matching algorithm, and it would empower those in research with a quicker starting point for testing modifications to ASW.

## Important Branches

1. **master**:  This git repo has several different attempts at optimizing the `asw.cu` cuda kernel.  The master branch has a mostly-stable snapshot of the cost-volume implementation of the ASW algorithm.  Note that there is a known memory issue with the `createCostVolume_kernel()` function in createCostVolume.cu, a bug which is fixed on the cost_volume branch but hasn't been backported.

2. **cost_volume**: The effort to use the cost-volume approach ("Fast cost-volume filtering for visual correspondence and beyond" by Rhemann et al) is developed on this branch.  Currently, the conversion from using a custom `struct cost_volume_t` to the more useful `cv::cuda::GpuMat` object is not complete, so it is not yet on the master branch.

## Current State

The cost-volume filtering method appears to have a higher minimum run time but ports much better to embedded hardware (1.6 sec runtime instead of 6.8 sec).

Currently the fastest attempt at GPU-acceleration exists with the old implementation (asw.cu) and can be tested by running `git checkout 9b87bdd` then `./asw l.png r.png 64 5 50`.

## Known Issues with initial implementation (asw.cu):

1. There seems to be some salt-and-pepper noise on the disparity output that I can't explain.

2. Shared memory is not handled well.  Currently shared memory size limits the combinations of numbers of disparities & window sizes available, but with a good implementation the size of shared memory should not offer any limit to those factors.  In fact, an attempt at reducing shared memory exists on the `dev` branch, but it actually made the shared memory issue worse.

3. I suspect improved performance could be achieved by tuning the auto-calculation of window size vs spacial sigma.

4. The current pixel-matching function is a sum of absolute difference (SAD), but a complete implementation should use a truncated absolute difference of cost and gradient (TAD C+G) as in the ASW paper.  An attempt at implementing TAD C+G can be found on the `half` branch, in the cpu version of the code.  Furthermore, modifications to the matching should be included to take into account sub-pixel disparities, such as in the paper, "A pixel dissimilarity measure that is insensitive to image sampling" by Birchfield and Tomasi.  However, such modifications are a lesser priority to issues such as the shared memory handling in CUDA.

5. Left and right disparity calculation comparison should be done.  Currently, only the left disparity is calculated.

# Known issues with cost-volume implementation:

1. This list isn't ready yet...
