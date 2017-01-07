/**
 * Primality Testing with CUDA (Fall 2016): 
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements an algorithm to test for primality discussed in the methodology section
 * of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the kernel call is_prime(int n, char *factor, char *prime)
 * to perform a parallel search for some factor of the value n.  The kernel calls are 
 * seperated into r=20 streams amongst the multi-stream processors of the CUDA compatible GPU.
 * This allows us to gather data via power analysis to find a relationship between 
 * execution speed and power dissipation for the Jetsion TK1.
 *
 * While the overhead of executing executing r streams slows down execution time,
 * the performance of the parallel search itself is significantly faster than it's
 * serial counterpart. We can see a significant improvement in the output displayed during runtime
 * when r = 1.
 * 
 * The output of the performance is displayed in seconds for verification.
 *
 * References:
 * NVIDIA CUDA C Programming Guide Version 3.2
 */

// System includes
#include <stdio.h>
#include <time.h>

// Jetson TK1 has device capability 1.x allowing 1024 threads/block
#define THREADS_PER_BLOCK 1024

// Performs a parallel search for a factor of the value n
// When a multiple is found, prime is written to 1 and facter
// is written as the multiple to be read & verified by the caller
//
// The values are written to device memory and must be recovered by the caller
__global__ void is_prime(int n, int *d_factor, int *d_prime) {
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i > 1 && i < n && n % i == 0) {
		*d_prime = 0;
		*d_factor = i;
	}
}

int main(void) {
	//r can be modified to produce as much overhead as needed during testing
	int *prime, *d_prime, n=900000006, r=20, *factor, *d_factor;
	cudaError_t error;

	/* Generate space on the device */
	prime = (int *)calloc(1, sizeof(int));
	*prime = 1;
	cudaMalloc((void **)&d_prime, sizeof(int));
	cudaMemcpy(d_prime, prime, sizeof(int), cudaMemcpyHostToDevice);
	factor = (int *)calloc(1, sizeof(int));
	cudaMalloc((void **)&d_factor, sizeof(int));

	/* Launch encrypt() kernel on GPU */
	cudaStream_t stream[r];
	for (int i = 0; i < r; i++ )
		cudaStreamCreate(&stream[i]);

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

    	// Execute the kernel
    	// NEED TO PUT STREAMS FOR R VALUE IN HERE
	for( int i = 0; i < r; i++){
		is_prime<<<(n + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK,0,stream[i]>>>(n, d_factor, d_prime);
		cudaStreamSynchronize(stream[i]);	
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

    	// Compute and print the performance
    	float msecPerisPrime = msecTotal / 1;

    	printf( "Performance= %.06f sec\n", msecPerisPrime/1000.0 );
	/*******************************************************************************************
	 ****************************** for testing purposes ***************************************/

	/* Destroy streams */
	for (int j = 0; j < r; j++){
    		cudaStreamDestroy(stream[j]);
  	}

	/* Copy results back to host */
	error = cudaMemcpy(prime, d_prime, sizeof(int), cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
    	{
        	printf("cudaMemcpy (prime,d_prime) returned error code %d, line(%d)\n", error, __LINE__);
        	exit(EXIT_FAILURE);
    	}

	error = cudaMemcpy(factor, d_factor, sizeof(int), cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
    	{
        	printf("cudaMemcpy (factor,d_factor) returned error code %d, line(%d)\n", error, __LINE__);
        	exit(EXIT_FAILURE);
    	}


	/*    IS IT PRIME???	*/
	if (*prime == 1) {
		printf("%d is prime.\n", n);
	} else {
		printf("%d is NOT prime, %d is a factor!\n", n, *factor);
	}


	/* Cleanup */
	free(prime); free(factor);
	cudaFree(d_prime); cudaFree(d_factor);
	
	return 0;
}
