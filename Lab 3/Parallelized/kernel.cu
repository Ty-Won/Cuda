#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#define eta 0.0002
#define rho 0.5
#define boundary_gain 0.75

#define BLOCKS 4
#define THREADS_PER_BLOCK 4
#define MATRIX_DIM 4

_global_ void simulate_inner(float* u, float* u1, float* u2) {

	unsigned int position = (threadIdx.x + (blockIdx.x * blockDim.x));

	if (((blockIdx.x > 0) && (blockIdx.x < blockDim.x-1)) && (threadIdx.x > 0 && threadIdx.x < blockDim.x-1)) {
		u[position] = (rho * (u1[position - blockDim.x] + u1[position + blockDim.x] + u1[position - 1] + u1[position + 1] - 4 * u1[position]) + 2 * u1[position] - (1 - eta) * (u2[position])) / (1 + eta);
	}

}

_global_ void simulate_borders(float* u) {
	//borders
	//row 0 and MATRIX_DIM
	unsigned int position = (threadIdx.x + (blockIdx.x * blockDim.x));


	if (blockIdx.x == 0 && (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1)) {
		//top border
		u[position] = boundary_gain * u[blockDim.x + position];
	}
	else if (blockIdx.x == blockDim.x - 1 && (threadIdx.x != 0 && threadIdx.x != blockDim.x - 1)) {
		//bottom border
		u[position] = boundary_gain * u[position - blockDim.x];
	}
	else if (threadIdx.x == 0 && (blockIdx.x > 0 && blockIdx.x < blockDim.x - 1)) {
		//Left border
		u[position] = boundary_gain * u[position + 1];
	}
	else if (threadIdx.x == blockDim.x - 1 && (blockIdx.x > 0 && blockIdx.x < blockDim.x - 1)) {
		//right border
		u[position] = boundary_gain * u[position - 1];
	}
}


_global_ void simulate_corners(float* u) {
	//corners
	//Top left
	u[0] = boundary_gain * u[MATRIX_DIM];
	// 	printf("top left: %f\n", u[0]);

		//Top right
	u[MATRIX_DIM - 1] = boundary_gain * u[MATRIX_DIM - 2];
	// 	printf("top right: %f\n", u[MATRIX_DIM - 1]);

		//Bottom left
	u[MATRIX_DIM * (MATRIX_DIM - 1)] = boundary_gain * u[MATRIX_DIM * (MATRIX_DIM - 2)];
	// 	printf("Bottom left: %f\n", u[MATRIX_DIM*(MATRIX_DIM - 1)]);

		//Bottom right
	u[MATRIX_DIM * (MATRIX_DIM - 1) + (MATRIX_DIM - 1)] = boundary_gain * u[MATRIX_DIM * (MATRIX_DIM - 1) + (MATRIX_DIM - 2)];
	// 	printf("bottom right: %f\n", u[MATRIX_DIM*(MATRIX_DIM-1) + (MATRIX_DIM-1)]);

}




int main(int argc, char* argv[])
{

	// 	unsigned int num_iterations = atoi(argv[3]);
	unsigned int num_iterations = 3;
	unsigned error;

	//Assigning Host Memory
	printf("Size of space: %d\n", MATRIX_DIM);
	size_t grid_size = MATRIX_DIM * MATRIX_DIM * sizeof(float);
	float* u = (float*)calloc(MATRIX_DIM * MATRIX_DIM, grid_size);
	float* u1 = (float*)calloc(MATRIX_DIM * MATRIX_DIM, grid_size);
	float* u2 = (float*)calloc(MATRIX_DIM * MATRIX_DIM, grid_size);

	//Assigning device memory
	float* cuda_u, * cuda_u1, * cuda_u2;
	cudaError_t cuda_status;

	cuda_status = cudaMalloc((void**)& cuda_u, grid_size);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cuda_status = cudaMemcpy(cuda_u, u, grid_size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}




	u1[((MATRIX_DIM * MATRIX_DIM) / 2 + MATRIX_DIM / 2)] = 1;

	cuda_status = cudaMalloc((void**)& cuda_u1, grid_size);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cuda_status = cudaMemcpy(cuda_u1, u1, grid_size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}


	cuda_status = cudaMalloc((void**)& cuda_u2, grid_size);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cuda_status = cudaMemcpy(cuda_u2, u2, grid_size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return -1;
	}


	printf("Elements: %f\n", u1[(MATRIX_DIM * MATRIX_DIM / 2) + MATRIX_DIM / 2]);

	int blocks = BLOCKS;
	int threads_per_block = THREADS_PER_BLOCK;

	clock_t start = clock();
	for (int iterations = 0; iterations < num_iterations; iterations++) {
		simulate_inner<<< BLOCKS, THREADS_PER_BLOCK >> >(cuda_u, cuda_u1, cuda_u2);
		cudaDeviceSynchronize();
		simulate_borders<<< BLOCKS, THREADS_PER_BLOCK >> >(cuda_u);
		cudaDeviceSynchronize();
		simulate_corners<<< BLOCKS, THREADS_PER_BLOCK >> >(cuda_u);


		cuda_status = cudaMemcpy(u, cuda_u, grid_size, cudaMemcpyDeviceToHost);
		printf("Element[N/2,N/2] for one node per thread: %f\n", u[(MATRIX_DIM * (MATRIX_DIM / 2)) + MATRIX_DIM / 2]);

		//cuda_status = cudaMemcpy(u1, cuda_u1, grid_size, cudaMemcpyDeviceToHost);
		//printf("Element[N/2,N/2]: %f\n", u1[(MATRIX_DIM * (MATRIX_DIM / 2)) + MATRIX_DIM / 2]);

		//cuda_status = cudaMemcpy(u2, cuda_u2, grid_size, cudaMemcpyDeviceToHost);
		//printf("Element[N/2,N/2]: %f\n", u2[(MATRIX_DIM * (MATRIX_DIM / 2)) + MATRIX_DIM / 2]);


		//Update u2
		float* u2_cuda_reference_p = cuda_u2;
		cuda_u2 = cuda_u1;
		cudaFree(u2_cuda_reference_p);

		//update u1
		cuda_u1 = cuda_u;
		cuda_status = cudaMalloc((void**)& cuda_u, grid_size);


	}


	free(u);
	free(u1);
	free(u2);

	cudaFree(cuda_u);
	cudaFree(cuda_u1);
	cudaFree(cuda_u2);

	return 0;
}