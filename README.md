# Adaptive Support Weight (ASW) correspondence matching

This git project is an open-source CUDA implementation of the algorithm described in "Adaptive support-weight approach for correspondence" by Kuk-Jin Yoon and In So Kweon, which is the basis for the most effective local reasoning stereo vision algorithms produced today.

## Goals and Motivations

To my knowledge, there is no other open-source implementation of ASW, although it was developed in 2005.  Freely-available computer vision libraries (such as OpenCV or Nvidia VisionWorks) do not offer ASW, either in CPU form or a GPU accelerated form.  The goal of this project is to implement a GPU-accelerated ASW algorithm which can ultimately be contributed to the OpenCV library.  A Free and Open Source Software (FOSS) implementation of ASW would empower those in industry with a more powerful stereo matching algorithm, and it would empower those in research with a quicker starting point for testing modifications to ASW.

## Branches

1. **master**:  This git repo has several different attempts at optimizing the `asw.cu` cuda kernel.  The master branch has what is currently the fastest attempt (for a discrete GPU), but **it is not on the most recent commit**.  Try running `git checkout 9b87bdd` and testing the code on your hardware!

2. **dev**: The most recent attempt at CUDA optimization lives on this branch.  The goal is for future development of the CUDA kernel to be done on this branch and merged into the master branch.

2. **half**: This branch has a pthread-accelerated CPU-based implementation of the ASW algorithm.  It is called 'half' because it was developed for testing the technique of using the asymmetric support-weight calculation method described in "Secrets of adaptive support-weight for local stereo matching" by Hosni et. al, where you only calculate the support weight for the reference window.  The utility of the CPU implementation is that it offers a simple means of testing modifications to the algorithm (such as different cost functions, or perhaps replacing the bilateral filter with a guided filter, etc).

## Current State

Currently the fastest attempt at GPU-acceleration can be tested by running `git checkout 9b87bdd` then `make && ./a.out l.png r.png 64 5 50`.  If your discrete gpu is enabled through the `optimus` command, you can just run `make run` to execute the code.

## Known Issues

1. There seems to be some salt-and-pepper noise on the disparity output that I can't explain.

2. Shared memory is not handled well.  Currently shared memory size limits the combinations of numbers of disparities & window sizes available, but with a good implementation the size of shared memory should not offer any limit to those factors.  In fact, an attempt at reducing shared memory exists on the `dev` branch, but it actually made the shared memory issue worse.

3. I suspect improved performance could be achieved by tuning the auto-calculation of window size vs spacial sigma.

4. The current pixel-matching function is a sum of absolute difference (SAD), but a complete implementation should use a truncated absolute difference of cost and gradient (TAD C+G) as in the ASW paper.  An attempt at implementing TAD C+G can be found on the `half` branch, in the cpu version of the code.  Furthermore, modifications to the matching should be included to take into account sub-pixel disparities, such as in the paper, "A pixel dissimilarity measure that is insensitive to image sampling" by Birchfield and Tomasi.  However, such modifications are a lesser priority to issues such as the shared memory handling in CUDA.

5. Left and right disparity calculation comparison should be done.  Currently, only the left disparity is calculated.

