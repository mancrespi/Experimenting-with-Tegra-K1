#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define THREADS_PER_BLOCK 1024
#define THRESHOLD 67108864

__global__ void encrypt(int n, char *m, char *k, char *c){
	int j, i =  blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){
		for(j = 1; j <= 100; j++){
			 c[i] = m[i] ^ k[i];
		}
	}
}

int main(){
	/* Serial on Host */
	FILE *fp_m, *fp_k;
	char *m, *k, *c, ch; /* Host copies */
	char *d_m, *d_k, *d_c; /* Device copies */
	int s_m, s_k, j,i;
	float msecPerencrypt, msecTotal = 0.0f;
	//printf("setup ...\n");
	//get size of files to malloc data
	if (!(fp_m = fopen("../../file.txt", "r")))
		perror("failed to read message file\n");
	
	while( fscanf(fp_m,"%c",&ch) != EOF ){
		s_m++;
	}

	if (!(fp_k = fopen("../../key.txt", "r")))
		perror("failed to read key\n");
	
	while( fscanf(fp_k,"%c",&ch) != EOF ){
		s_k++;
	}
	
	//printf("mallocs cpu...\n");

	//malloc space for m, k, c
	if ( !(m = (char *)malloc(sizeof(char)*s_m)) ){ 
		printf("Failed on malloc for m\n");
		exit(EXIT_FAILURE);
	}

	if ( !(k = (char *)malloc(sizeof(char)*s_k)) ){ 
		printf("Failed on malloc for k\n");
		exit(EXIT_FAILURE);
	}

	if ( !(c = (char *)malloc(sizeof(char)*s_m)) ){ 
		printf("Failed on malloc for c\n");
		exit(EXIT_FAILURE);
	}

	/* Alloc space for device copies of m, k, c */
	cudaError_t error;
	printf("mallocs gpu...\n");
	error = cudaMalloc((void **)&d_m, s_m);
	error = cudaMalloc((void **)&d_k, s_k);
	error = cudaMalloc((void **)&d_c, s_m);

	fseek(fp_m, 0, 0);
	fseek(fp_k, 0, 0);

	//read into buffers
	printf("read data...\n");
	for( j = 0; fscanf(fp_m,"%c",&ch) != EOF; j++ ){
		m[j] = ch;
	}
	for( j = 0; fscanf(fp_k,"%c",&ch) != EOF; j++ ){
		k[j] = ch;
	}

	/* Copy inputs to device */
	printf("Copy to device...\n");
 	cudaMemcpy(d_m, m, s_m, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_k, k, s_k, cudaMemcpyHostToDevice);
	
	printf("Setting up streams...\n");
	int sections = s_m/THRESHOLD;
	int rem = s_m%THRESHOLD;
	cudaStream_t stream[sections];
	
	for(i = 0; i < sections; i++){
		cudaStreamCreate(&stream[i]);	
	}
	printf("moving on...\n");

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

	/* Parallel on Device */
 	/* Launch encrypt() kernel on GPU with N blocks */
	for(i = 0; i < sections-1; i++ ){
 		encrypt<<<(s_m+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(s_m, d_m, d_k, d_c);
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

    	//float msecTotal = 0.0f;
  
	/* Copy result back to host */
 	error = cudaMemcpy(c, d_c, s_m, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
    	{
        	printf("cudaMemcpy (c,d_c) returned error code %d, line(%d)\n", error, __LINE__);
        	exit(EXIT_FAILURE);
    	}

	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        	exit(EXIT_FAILURE);
    	}

	msecPerencrypt = msecTotal / 1;
    	printf( "Performance= %.06f sec\n", msecPerencrypt/1000.0 );

	//validate for correctness
 	for (j = 0; j < s_m; j++){
		if( c[j] != (m[j]^k[j]) ){
			printf("WRONG! c[%d] != m[%d]^k[%d] ==> c='%c',m^k=%c\n", j,j,j,c[j],m[j]^k[j]);
			//exit(EXIT_FAILURE);
		}
	}

	// Compute and print the performance
    	//float msecPerencrypt = msecTotal / 1;
	
	/* Cleanup */
	/* Destroy streams */
	//for (j = 0; j < sections; j++){
    		//cudaStreamDestroy(stream[j]);
  	//}

 	free(m); free(k); free(c);
 	cudaFree(d_m); cudaFree(d_k); cudaFree(d_c);
 	fclose(fp_m); fclose(fp_k);

	return 0;
}
