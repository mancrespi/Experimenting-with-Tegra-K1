/**
 * Vector Addition with CUDA (Fall 2016):
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements a parallel vector addition: C = A + B
 * discussed in the methodology section of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the function call vectorAdd( int n, int *A, int *B, int *C )
 * to perform a vector addition of two arrays A and B of length n 
 *
 * The output is verified before the program terminates to see that every 
 * element of index i has C[i] == A[i] + B[i]
 *
 * We can see that there is a significant speedup in comparison to the time it takes
 * to perform the vectorAdd in the serial code.
 *
 * The output of the performance is displayed in seconds. 
 * The runtime results are to be compared with the performance of vector.c
 *
 */

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Jetson TK1 has device capability 1.x allowing 1024 threads/block
// defines dimension of vectored data as XDIM
#define XDIM 413124 
#define THREADS_PER_BLOCK 1024

// Performs a vector addition of arrays v1 and v2
// in parallel on the GPU.  The index i is specified as 
// a particular thread within some block on the device
// This allows for parallel computation on the device
__global__ void vectorAdd( int n, int *v1, int *v2, int *v ){
	int i =  blockIdx.x*blockDim.x + threadIdx.x;
	if( i < n ){ 
		v[i] = v1[i] + v2[i];
	}	
}

int main(void){
	int i, v1[XDIM], v2[XDIM], v[XDIM];
        int *d_v1, *d_v2, *d_v, size = sizeof(int)*XDIM; 
	cudaError_t error;

	//Populate data in v1 and v2
	for( i = 0; i < XDIM; i++ ){
		v1[i] = (i % 5)+1; 
		v2[i] = (i % 5)+1;
	}

	//set up space on GPU device
	cudaMalloc( (void **) &d_v1, size );
	cudaMalloc( (void **) &d_v2, size );
	cudaMalloc( (void **) &d_v, size );

	//write data to GPU device
	cudaMemcpy( d_v1, v1, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( d_v2, v2, size, cudaMemcpyHostToDevice ); 
	/*******************************for testing purposes****************************************
	 *******************************************************************************************/
	cudaDeviceSynchronize();

    	// Allocate CUDA events that we'll use for timing
    	cudaEvent_t start;
    	error = cudaEventCreate(&start);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

    	cudaEvent_t stop;
    	error = cudaEventCreate(&stop);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

    	// Record the start event
    	error = cudaEventRecord(start, NULL);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

 	vectorAdd<<<(XDIM+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( XDIM, d_v1, d_v2, d_v );
	
	// Record the stop event
    	error = cudaEventRecord(stop, NULL);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

    	// Wait for the stop event to complete
    	error = cudaEventSynchronize(stop);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

    	float msecTotal = 0.0f;
    	error = cudaEventElapsedTime(&msecTotal, start, stop);

    	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

    	// Compute and print the performance
    	float msecPerVecAdd = msecTotal / 1;

    	printf( "Performance= %.06f sec\n", msecPerVecAdd/1000.0 );
	/*******************************************************************************************
	 ****************************** for testing purposes ***************************************/
		

	//Retrieve data from GPU
	cudaMemcpy( v, d_v, size, cudaMemcpyDeviceToHost ); 
	
	//Validate for correctness
	for( i = 0; i < XDIM; i++ ){
		if( v[i] != v1[i] + v2[i] ){
			printf("WRONG!!\n");
			exit(EXIT_FAILURE);
		}
	}	
	
	return 0;
}
