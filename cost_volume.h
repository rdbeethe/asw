#ifndef COST_VOLUME_H
#define COST_VOLUME_H

struct cost_volume_t {
	float* volume;
	int nrows;
	int ncols;
	int ndisp;
	int stride;
};

struct rgba_pixel {
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

struct cost_volume_t get_gpu_volume(struct cost_volume_t vin);
void viewSlices(struct cost_volume_t& cost_volume, int first, int last);
void costVolumeBoxFilter(struct cost_volume_t& cost_volume, int kernelSize);

#endif // COST_VOLUME_H
