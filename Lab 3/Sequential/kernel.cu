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
	return 0;
}
