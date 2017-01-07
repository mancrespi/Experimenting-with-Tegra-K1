/**
 * Array Search with CUDA (Fall 2016): 
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements an array search discussed in the methodology section
 * of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the kernel call search(int n, char *data, char *out, char c)
 * to perform a parallel search for char c amongst a large vector of characters.
 *
 * The result is a one-one mapping of the data array with 1s and 0s in the out array at 
 * corresponding indices where the character has been found.  The output is written to 
 * the file 'parallel_array_search_result.txt' for validation with 'array_search_result.txt'
 * when running the executable for array_search.c
 *
 * While the overhead of executing 'cudaMemCpy(...)' slows down execution time,
 * the performance of the parallel search itself is significantly faster than it's
 * serial counterpart.
 * 
 * The output of the performance is displayed in seconds for verification.
 *
 * References:
 * NVIDIA CUDA C Programming Guide Version 3.2
 */

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

// Jetson TK1 has device capability 1.x allowing 1024 threads/block
// We also indicate a threshold of 67108864 for vectored data
#define THREADS_PER_BLOCK 1024 
#define THRESHOLD 67108864 

//Identify failures
#define FILE_OPEN_FAIL -1
#define MALLOC_FAIL -2


/* 
 * Indicates the task search to be performed on the GPU
 * for char c in array data both of size n
 *
 * The output 1 or 0 is written in out to indicate 'found' and 'not-found' respectively
 * Results are written to device memory and must be fetched back from out for verification.
 * search( n, ['a','b','d','c','d','e',...], result, 'd') ==> result = ['0','0','1','0','1','0',...]
 */
__global__ void search(int n, char *data, char *out, char c){
	int i =  blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n){
		if (data[i] == c){
			out[i] = '1';
		}else{
			out[i] = '0';
		}
	}
}

int main(){
	FILE *fp_data, *fp_out;
	char *data, c;
	char *d_data, *d_out;
	int s_data = 0, j = 0, i = 0;

	int flag = 0;
	cudaError_t error;
	//printf("Computing file size...\n");
	
	if (!(fp_data = fopen("../../file.txt", "r"))){
		perror("failed to open file.txt\n");
		return FILE_OPEN_FAIL;
	}
	
	while( fscanf(fp_data,"%c",&c) != EOF ){
		s_data++;
	}

	int rem = s_data % THRESHOLD;
	int sections = (THRESHOLD+s_data-1)/THRESHOLD;

	//printf("Mallocing %d bytes of data on CPU...\n", s_data);
	
	/* Allocate necessary space for host buffer */
	cudaMallocHost(&data, sizeof(char)*s_data);

	/* Allocate necessary space for device buffer */
	//printf("Mallocing %d bytes of data on GPU...\n", s_data);	
	cudaMalloc( (void **) &d_data, sizeof(char)*s_data);
	cudaMalloc( (void **) &d_out, sizeof(char)*s_data);

	fseek(fp_data, 0, 0);

	/* Read file into buffer */
	//printf("Reading data into buffer...\n");	
	for( j= 0; fscanf(fp_data,"%c",&data[j]) != EOF; j++ ){	}

	/* Identify our streams */
	cudaStream_t stream[sections];
	for (int j = 0; j < sections; j++){
		cudaStreamCreate(&stream[j]);
	}

	/* Time the search algorithm */
	//printf("Executing search on GPU...\n");


	if( rem == 0 ){
		flag = 0;
	}else{
		flag = 1;
	}
	
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
	printf("Running...\n");
    	// Execute the kernel
	for(j = 0; j < sections-flag; j++){
		cudaMemcpyAsync(d_data + j * THRESHOLD, data + j * THRESHOLD,
                    THRESHOLD, cudaMemcpyHostToDevice, stream[j]);
		search<<<(THRESHOLD+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK,0,stream[j]>>>(THRESHOLD, 
								data+(j*THRESHOLD), d_out+(j*THRESHOLD), 'D');
		cudaStreamSynchronize(stream[j]);	
		cudaMemcpyAsync( data + j * THRESHOLD, d_out + j * THRESHOLD,
                    THRESHOLD, cudaMemcpyDeviceToHost, stream[j]);
	}	
	
	
	/* Define and run stream for remainder */
	cudaMemcpyAsync(d_data + j * THRESHOLD, data + j * THRESHOLD,
                    rem, cudaMemcpyHostToDevice, stream[j]);

	search<<<(rem+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK,0,stream[j]>>>(rem, 
							data + j * THRESHOLD, d_out + j * THRESHOLD, 'D');
	cudaStreamSynchronize(stream[j]);	

	cudaMemcpyAsync( data + j * THRESHOLD, d_out + j * THRESHOLD,
                rem, cudaMemcpyDeviceToHost, stream[j]);

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
	for (int j = 0; j < sections; j++){
    		cudaStreamDestroy(stream[j]);
  	}

	/* Copy result back to host */
  	//printf("Writing result to 'parallel_array_search_result.txt'...\n");
 	
	if ( !(fp_out = fopen("parallel_array_search_result.txt", "w")) ){
		perror("failed to open results file\n");
		return FILE_OPEN_FAIL;
	}

	//output data to file
 	for (j = 0; j < s_data; j++){
		if( i == 32 ){
			fprintf(fp_out, "%c\n", data[j]);
			i = 0;	
		}else{
			fprintf(fp_out, "%c", data[j]);
			i++;
		}


	}

	/* Cleanup */
 	cudaFreeHost(data); 
	cudaFree(d_data); cudaFree(d_out);
 	fclose(fp_data); fclose(fp_out);

 	return 0;
}
