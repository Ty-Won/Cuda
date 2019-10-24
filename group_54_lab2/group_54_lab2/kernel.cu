
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "wm.h"
#include <stdio.h>
#include "lodepng.h"
#include <time.h>
#define MAX_THREAD 1024

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
const unsigned int img_width = 5;
const unsigned int img_height = 5;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[])
{

	// Inverse
	unsigned error;
	unsigned char* image, * new_image;
	unsigned int width, height;

	char* input_img = argv[1];
	unsigned int weight_dim = atoi(argv[2]);
	unsigned char** matrix;

	clock_t start = clock();


	error = lodepng_decode32_file(&image, &width, &height, input_img);
	error = lodepng_decode32(matrix, &width, &height, image, 1024);
	printf("value of c: %d\n",*image);







    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
  //  cudaStatus = cudaDeviceReset();
  //  if (cudaStatus != cudaSuccess) {
  //      fprintf(stderr, "cudaDeviceReset failed!");
  //      return 1;
  //  }

    return 0;
}


void getCofactor(int matrix[img_height][img_width], int temp[img_height][img_width], int row, int col) {
	int temp_row = 0, temp_col = 0;
	int row_size = sizeof(matrix) / sizeof(matrix[0]);
	int col_size = sizeof(matrix[0]) / sizeof(matrix[0][0]);

	for (int row = 0; row < row_size; row++) {
		for (int col; col < col_size; col++) {

			if (row != temp_row && col != temp_col) {
				temp[temp_row][temp_col++] = matrix[row][col];
				
				if (temp_col == col_size) {
					temp_row++;
					temp_col = 0;

				}
			}

		}
	}

}

int determinant(int matrix[img_height][img_width], int sub_dimension) {
	int determinant = 0;
	int temp[img_height][img_width];
	int sign = 1;

	if (sub_dimension == 1) {
		return matrix[0][0];
	}

	// Iterate for each element of first row 
	for (int col = 0; col < img_width; col++)
	{
		// Getting Cofactor of A[0][f] 
		getCofactor(matrix, temp,img_height, img_width);
		D += sign * matrix[0][col] * determinant(temp, img_height - 1);

		// terms are to be added with alternate sign 
		sign = -sign;
	}

	return D;


}





// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
