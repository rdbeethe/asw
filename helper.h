#ifndef HELPER_H
#define HELPER_H

__global__ void gpu_memset(unsigned char* start, unsigned char value, int length);
void gpu_perror(const char* input);

#endif // HELPER_H
