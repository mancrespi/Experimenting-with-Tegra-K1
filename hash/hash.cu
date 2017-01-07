/**
 * One Way Hash with CUDA (Fall 2016):
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements a simple hash from a space of size 2n --> n 
 *
 * The following code makes use of the kernel call hash(char *f, char *h, int n)
 * to perform a parallel hash of elements f --> h with corresponding indices 2i --> i 
 *
 * The result is a mapping of the data within f to the data within h 
 * The output is verified before the program terminates to see that every 
 * element at index 2i of f is indeed at index i in h
 *
 * We can see that there is a significant speedup in comparison to the time it takes
 * to perform the hash in the serial code.
 * 
 * The output of the performance is displayed in seconds. 
 * The performance results are to be compared with the performance of hash.c
 *
 */

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// Jetson TK1 has device capability 1.x allowing 1024 threads/block
// We also indicate EVEN_NUM as the vector size since this hash requires even length arrays
#define THREADS_PER_BLOCK 1024 
#define EVEN_NUM 123374234


__global__ void hash(char *f, char *h, int n) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if( i < n ){
		h[i] = f[2*i];	
	}
}

int main(void) {
	int two_n = EVEN_NUM, i, r=50;
	char *f, *h, *d_f,*d_h;
	cudaError_t error;

	if ( two_n % 2 ){
		printf("NO NO NO!!! Even numbers only please.\n");
		exit(EXIT_FAILURE);
	}

	//printf("Malloc space on CPU (f,h)");
	f = (char *)calloc(sizeof(char), two_n);

	if( f == NULL ){
		fprintf(stderr,"Failed to allocate %d bytes for f.",two_n);
		exit(EXIT_FAILURE);
	}

	h = (char *)calloc(sizeof(char), two_n/2);

	if( h == NULL ){
		fprintf(stderr,"Failed to allocate %d bytes for h.",two_n/2);
		exit(EXIT_FAILURE);
	}

	/* Identify our streams */
	//printf("Malloc space on GPU (d_f,d_h)\n");
	error = cudaMalloc((void **)&d_f, sizeof(char) * two_n);
	
	if( error != cudaSuccess ){
		fprintf(stderr,"Failed to cudaMalloc %d bytes for d_f.",two_n);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_h, sizeof(char) * two_n/2);
	
	if( error != cudaSuccess ){
		fprintf(stderr,"Failed to cudaMalloc %d bytes for d_h.",two_n/2);
		exit(EXIT_FAILURE);
	}

	//populate data into array
	//printf("Generate vectored data (Size=%d bytes)\n",two_n);
	for (i = 0; i < two_n; i++) {
		f[i] = (char) ((i % 94) + 33);
	}

	//send data over the bus
	//printf("Send data to GPU\n");
	error = cudaMemcpy( d_f, f, two_n, cudaMemcpyHostToDevice);
	
	if (error != cudaSuccess)
    	{
        	printf("cudaMemcpy (d_f,f) returned error code %d, line(%d)\n", error, __LINE__);
        	exit(EXIT_FAILURE);
    	}

	/*************************** Setup for testing ************************************/
	//printf("Run kernel code \n");


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
	printf("Running...\n");
	//run kernel
	for( i = 0; i < r; i++){
		hash<<<(two_n/2+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_f,d_h,two_n/2);
	}
	
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

    	
	/*******************************************************************************************
	 ****************************** for testing purposes ***************************************/

	//send data over the bus
	error = cudaMemcpy( h, d_h, sizeof(char)*two_n/2, cudaMemcpyDeviceToHost);
	
	if (error != cudaSuccess)
    	{
        	printf("cudaMemcpy (h,d_h) returned error code %d, line(%d)\n", error, __LINE__);
        	exit(EXIT_FAILURE);
    	}
	//printf("Done.\n");

	//validate for correctness
	for (i = 0; i < two_n/2; i++) {
		if (h[i] != f[2*i]) {
			//printf("index %d FAILED!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	// Compute and print the performance
    	float msecPerhash = msecTotal / 1;
    	printf( "Performance= %.06f sec\n", msecPerhash/1000.0 );

	free(f); free(h);
	cudaFree(d_f); cudaFree(d_h);
	
	cudaDeviceReset();

	return 0;
}
