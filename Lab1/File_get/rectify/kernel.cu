
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include "lodepng.h"
#define MAX_THREAD 1024

__global__ void rectify(unsigned char * original_img, unsigned char* new_img, unsigned int num_thread, unsigned int id, unsigned int size) {
	int i = threadIdx.x + id;
	if(i < size)
		new_img[i] = (original_img[i] < 127 ? 127 : original_img[i]);
}

int main(int argc, char *argv[]) {
	if (argc != 4) {
		printf("Invalid number of arguments\n");
		return -1;
	}
	clock_t start = clock();

	unsigned char* original_img, *new_img;
	unsigned char* original_cudaImg, *new_cudaImg;

	unsigned int num_thread = atoi(argv[3]);
	if (num_thread > 1024) {
		printf("Maximum number of threads allowed is 1024. %d is too big\n", num_thread);
		return -1;
	}
	unsigned width, height;
	unsigned int imagesize;
	unsigned error;
	error = lodepng_decode32_file(&original_img, &width, &height,
		argv[1]);
	if (error) {
		printf("%d: %s\n", error, lodepng_error_text(error));
		return -1;
	}

	imagesize = width * height * 4 * sizeof(unsigned char);
	new_img = (unsigned char*)malloc(imagesize);

	cudaMalloc((void**)&original_cudaImg, imagesize);
	cudaMalloc((void**)&new_cudaImg, imagesize);
	cudaMemcpy(original_cudaImg, original_img, imagesize, cudaMemcpyHostToDevice);
	for (unsigned int i = 0; i < imagesize; i = i + num_thread) {
		rectify<<<1,num_thread>>>(original_cudaImg, new_cudaImg, num_thread, i, imagesize);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(new_img, new_cudaImg, imagesize, cudaMemcpyDeviceToHost);

	error = lodepng_encode32_file(argv[2], new_img, width, height);
	if (error) {
		printf("%d: %s\n", error, lodepng_error_text(error));
		return -1;
	}
	printf("%lu msec", clock() - start);

	cudaFree(original_cudaImg);
	cudaFree(new_cudaImg);
	
	return 0;
}