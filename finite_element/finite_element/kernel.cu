
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 0.0002
#define P 0.5
#define G 0.75

#define BLOCKS 1024
#define THREADS_PER_BLOCK 16

//Assume square matrix
#define MATRIX_DIM 4

__global__ void matrix_init(float *matrix, unsigned int tasks, unsigned int matrix_size) {
	unsigned int id = (threadIdx.x + (blockIdx.x * blockDim.x))*tasks;
	unsigned int position;
	for (unsigned int i = 0; i < tasks && (i+id) < matrix_size; ++i) {
		position = i + id;
		matrix[position] = 0;
	}
}

__global__ void hit_the_drum(float* matrix, int x, int y) {
	matrix[x + y * MATRIX_DIM] = 1;
}

__global__ void simulate(float* u0, float* u1, float* u2, unsigned int tasks, unsigned int matrix_size) {
	unsigned int id = (threadIdx.x + (blockIdx.x * blockDim.x)) * tasks;
	unsigned int position;
	unsigned int special_case = 0;
	for (unsigned int i = 0; (i < tasks) && ((i + id) < matrix_size); ++i) {
		position = i + id;

		// All the outer elements should be handled later.
		////Check for corner:
		//if (position == 0) { //(0,0)
		//	//u0[0] = G * u0[1];
		//}
		//else if (position == MATRIX_DIM - 1) { //(N-1,0)
		//	//u0[MATRIX_DIM - 1] = G * u0[MATRIX_DIM - 2];
		//}
		//else if (position == ((MATRIX_DIM - 1) * MATRIX_DIM) || position == (MATRIX_DIM-1 + (MATRIX_DIM - 1) * MATRIX_DIM)) {//(0,N-1),(N-1,N-1)
		//	//u0[position] = G * u0[position - MATRIX_DIM];
		//}

		////Check for side:
		//else if ((position % MATRIX_DIM) == 0) {//(0,i)
		//	//u0[position] = G * u0[position + 1];
		//}
		//else if ((position % MATRIX_DIM) == (MATRIX_DIM - 1)) { // (N-1,i)
		//	//u0[position] = G * u0[position - 1];
		//}
		//else if (position < MATRIX_DIM) {//(i,0)
		//	//u0[position] = G * u0[position + MATRIX_DIM];
		//}
		//else if (position >= (MATRIX_DIM - 1) * MATRIX_DIM) {//(i,N-1)
		//	//u0[position] = G * u0[position - MATRIX_DIM];
		//}

		////Inner elements
		//else {
		if(position%MATRIX_DIM > 0 && position%MATRIX_DIM < MATRIX_DIM-1 && position > MATRIX_DIM && position < (MATRIX_DIM-1)*MATRIX_DIM){
			u0[position] = P * (u1[position - 1] + u1[position + 1] + u1[position - MATRIX_DIM] + u1[position + MATRIX_DIM] - 4 * u1[position])
				+ 2 * u1[position] - (1 - N) * u2[position];
			u0[position] = u0[position] / (1 + N);
			//printf("[position]: %f\n", position, u0[position]);
		}

	}
}

__global__ void simulate_sides(float* u0, unsigned int tasks, unsigned int matrix_size) {
	unsigned int id = (threadIdx.x + (blockIdx.x * blockDim.x)) * tasks;
	unsigned int position;
	unsigned int special_case = 0;
	for (unsigned int i = 0; (i < tasks) && ((i + id) < matrix_size); ++i) {
		position = i + id;
		//Check for corner, which will be computed later.
		if (position == 0) { //(0,0)
			//u0[0] = G * u0[1];
		}
		else if (position == MATRIX_DIM - 1) { //(N-1,0)
			//u0[MATRIX_DIM - 1] = G * u0[MATRIX_DIM - 2];
		}
		else if (position == ((MATRIX_DIM - 1) * MATRIX_DIM) || position == (MATRIX_DIM-1 + (MATRIX_DIM - 1) * MATRIX_DIM)) {//(0,N-1),(N-1,N-1)
			//u0[position] = G * u0[position - MATRIX_DIM];
		}

		//Check for side
		else if ((position % MATRIX_DIM) == 0) {//(0,i)
			u0[position] = G * u0[position + 1];
		}
		else if ((position % MATRIX_DIM) == (MATRIX_DIM - 1)) { // (N-1,i)
			u0[position] = G * u0[position - 1];
		}
		else if (position < MATRIX_DIM) {//(i,0)
			u0[position] = G * u0[position + MATRIX_DIM];
		}
		else if (position >= (MATRIX_DIM - 1) * MATRIX_DIM) {//(i,N-1)
			u0[position] = G * u0[position - MATRIX_DIM];
		}

	}
}

__global__ void simulate_corners(float* u0, unsigned int tasks, unsigned int matrix_size) {
	unsigned int id = (threadIdx.x + (blockIdx.x * blockDim.x)) * tasks;
	unsigned int position;
	unsigned int special_case = 0;
	for (unsigned int i = 0; (i < tasks) && ((i + id) < matrix_size); ++i) {
		position = i + id;
		//Check for corner:
		if (position == 0) { //(0,0)
			u0[0] = G * u0[1];
		}
		else if (position == MATRIX_DIM - 1) { //(N-1,0)
			u0[MATRIX_DIM - 1] = G * u0[MATRIX_DIM - 2];
		}
		else if (position == ((MATRIX_DIM - 1) * MATRIX_DIM) || position == (MATRIX_DIM - 1 + (MATRIX_DIM - 1) * MATRIX_DIM)) {//(0,N-1),(N-1,N-1)
			u0[position] = G * u0[position - MATRIX_DIM];
		}

	}
}

void shift_reference(float* &u0, float* &u1, float* &u2) {
	cudaFree(u2);
	u2 = u1;
	u1 = u0;
}

int main(int argc, char* argv[]) {

	if (argc != 2) {
		printf("Please enter a positive number of iteration as the argument.\n");
		return -1;
	}
	unsigned int iteration = atoi(argv[1]);

	float* u0;

	float* cuda_u0, *cuda_u1, * cuda_u2;
	float* cuda_dummy; 

	size_t matrix_size_float = MATRIX_DIM * MATRIX_DIM * sizeof(float);
	unsigned int matrix_size = MATRIX_DIM * MATRIX_DIM;
	cudaError_t cudaStatus;



	//malloc u0;
	u0 = (float*)malloc(matrix_size_float);



	//cudaMalloc
	cudaStatus = cudaMalloc((void**)&cuda_u0, matrix_size_float);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&cuda_u1, matrix_size_float);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&cuda_u2, matrix_size_float);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	//Initialize number of tasks per threads.
	unsigned int tasks_thread;
	if(matrix_size% (BLOCKS * THREADS_PER_BLOCK) == 0)
		tasks_thread = matrix_size / (BLOCKS * THREADS_PER_BLOCK);
	else
		tasks_thread = matrix_size / (BLOCKS * THREADS_PER_BLOCK) + 1;
	
	
	//Initialize matrix to 0
	matrix_init << <BLOCKS, THREADS_PER_BLOCK >> > (cuda_u1, tasks_thread, matrix_size);
	matrix_init << <BLOCKS, THREADS_PER_BLOCK >> > (cuda_u2, tasks_thread, matrix_size);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}

	hit_the_drum << <1, 1 >> > (cuda_u1, MATRIX_DIM/2, MATRIX_DIM/2);

	cudaDeviceSynchronize();

	//Simulation
	for (int i = 0; i < iteration; ++i) {
		simulate << <BLOCKS, THREADS_PER_BLOCK >> > (cuda_u0, cuda_u1, cuda_u2, tasks_thread, matrix_size);
		simulate_sides << <BLOCKS, THREADS_PER_BLOCK >> > (cuda_u0, tasks_thread, matrix_size);
		simulate_corners << <BLOCKS, THREADS_PER_BLOCK >> > (cuda_u0, tasks_thread, matrix_size);
		cudaDeviceSynchronize();
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return -1;
		}

		shift_reference(cuda_u0, cuda_u1, cuda_u2);

		//cudaMalloc
		cudaStatus = cudaMalloc((void**)&cuda_u0, matrix_size_float);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return -1;
		}

		cudaStatus = cudaMemcpy(u0, cuda_u1, matrix_size_float, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return -1;
		}
	}
	printf("[%d][%d]: %.5f\n", MATRIX_DIM/2, MATRIX_DIM/2, u0[MATRIX_DIM/2 + (MATRIX_DIM/2)*MATRIX_DIM]);

	printf("Alles ist gut\n");
	cudaFree(cuda_u0);
	cudaFree(cuda_u1);
	cudaFree(cuda_u2);
	free(u0);
	return 0;
}