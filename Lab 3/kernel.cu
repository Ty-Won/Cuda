// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#define MAX_THREAD 1024




int main(int argc, char* argv[])
{

	float eta = 0.0002;
	float rho = 0.5;
	float boundary_gain = 0.75;
	int N = 4;

// 	unsigned int num_iterations = atoi(argv[3]);
	unsigned int num_iterations = 3;
	unsigned error;



	printf("Size of space: %d\n", N);
	size_t grid_size = N * N * sizeof(float);
	float* u = (float*)calloc(N*N, grid_size);
	float* u1 = (float*)calloc(N*N, grid_size);
	float* u2 = (float*)calloc(N*N, grid_size);




	u1[((N * N)/ 2 + N / 2)] = 1;
	
	printf("Elements: %f\n", u1[(N*N/2)+N/2]);

	clock_t start = clock();
	for (int iterations = 0; iterations < num_iterations; iterations++) {
	
    	//Inner
    	for (int u_row = 1; u_row <= N-2; u_row++) {
    		for (int u_col = 1; u_col <= N-2; u_col++) {
    			u[N * u_row + u_col] = (rho * (u1[N * (u_row-1) + u_col] + u1[N * (u_row + 1) + u_col] + u1[u_row * N + (u_col - 1)] + u1[(N *u_row) + u_col +1] - 4 * u1[N * u_row + u_col]) + 2 * u1[N * u_row + u_col] - (1 - eta) * (u2[N * u_row + u_col]))/(1+eta);
    		    printf("Elements: %f\n", u[N * u_row + u_col]);

    		    
    		}
    	}
        
    	
        //borders
    	//row 0 and N
    	for (int u_col=1; u_col <= N-2; u_col++){
    	    //row 0
    	    u[u_col] = boundary_gain*u[N+u_col];
    	    
    	    //row N
    	    u[(N*(N-1))+u_col] = boundary_gain*u[N*(N-2)+u_col];
    	    
    	}
    	
    	//Col 0 and N
    	//row N
        for (int u_row = 1; u_row <= N-2; u_row++){
            // Col 0
    	    u[u_row*N] = boundary_gain*u[N*u_row+1];
    	    
    	   // Col N
    	    u[(u_row*N)+N-1] = boundary_gain*u[(N*u_row) + (N-2)];
    	}
    	
        //corners
        //Top left
    	u[0] = boundary_gain * u[N];
    // 	printf("top left: %f\n", u[0]);
    	
    	//Top right
    	u[N - 1] = boundary_gain * u[N - 2];
    // 	printf("top right: %f\n", u[N - 1]);
    	
    	//Bottom left
    	u[N*(N - 1)] = boundary_gain * u[N*(N - 2)];
    // 	printf("Bottom left: %f\n", u[N*(N - 1)]);
    	
    	//Bottom right
    	u[N*(N-1) + (N-1)] = boundary_gain * u[N*(N-1) + (N-2)];
    // 	printf("bottom right: %f\n", u[N*(N-1) + (N-1)]);
        
        
        
        printf("Element[N/2,N/2]: %f\n", u[(N*(N/2))+N/2]);
        
        
        
    	//Update u2
    	float* u2_reference_p = u2;
    	u2 = u1;
    	free(u2_reference_p);
    
    	//update u1
    	u1 = u;
    	u = (float*)calloc(N * N, grid_size);
    	
    	
	
	}
	
    
    free(u);
    free(u1);
    free(u2);




	////Allocate device memory
	//float* cuda_grid;
	//cudaMalloc((void**)& cuda_grid, grid_size);;


	//cudaMemcpy(cuda_grid, grid, grid_size, cudaMemcpyHostToDevice);

	//int num_blocks = (int)ceil((out_img_height * out_img_width) / num_thread);
	//int threads_per_block = (int)ceil(num_thread / num_blocks);
	//printf("num_thread: %d\n", num_thread);
	//printf("threads_per_block: %d\n", threads_per_block);
	//for (unsigned int i = 0; i < out_img_size; i = i + num_thread) {
	//	simu.late << <1, num_thread >> > (cuda_original_img, img_width, img_height, cuda_out_img, c_w_matrix, weight_side_len, i, num_thread);
	//}
	//cudaDeviceSynchronize();
	//cudaMemcpy(out_img, cuda_out_img, out_img_size, cudaMemcpyDeviceToHost);

	//error = lodepng_encode32_file(argv[2], out_img, out_img_width, out_img_height);
	//if (error) {
	//	printf("%d: %s\n", error, lodepng_error_text(error));
	//	return -1;
	//}
	//printf("%ul msec", clock() - start);

	//free(original_img);
	//free(out_img);

	//cudaFree(cuda_original_img);
	//cudaFree(cuda_out_img);

	return 0;
}
