/**
 * Vector Addition Test Serial (Fall 2016):
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements a sequential vector addition: C = A + B
 * discussed in the methodology section of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the function call vectorAdd( int n, int *A, int *B, int *C )
 * to perform a vector addition of two arrays A and B of length n 
 *
 * The output is verified before the program terminates to see that every 
 * element of index i has C[i] == A[i] + B[i]
 *
 * The output of the performance is displayed in seconds. 
 * The runtime results are to be compared with the performance of vector.cu
 *
 */

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h> 

//dimension for vectored data
#define XDIM 413124

// Performs a sequential vector addition of two arrays v1 and v2 of length n
// The result of the addition is left within the corresponding indices 
// of v: v[i] = v1[i] + v2[i]
vectorAdd( int n, int *v1, int *v2, int *v ){
	int i;
	for( i = 0; i < n; i++){
		v[i] = v1[i] + v2[i];
	}	
}

int main(void){
	//XDIM can be modified to perform larger additions
	int i, v1[XDIM], v2[XDIM], v[XDIM];
	struct timeval tv_start, tv_stop, tv_diff; 

	//Populate data into arrays v1 and v2
	for( i = 0; i < XDIM; i++ ){
		v1[i] = (i % 5)+1; 
		v2[i] = (i % 5)+1;
	}

	//run test
	gettimeofday(&tv_start, NULL);
 	vectorAdd( XDIM, v1, v2, v );				
	gettimeofday(&tv_stop, NULL);
	timersub(&tv_stop, &tv_start, &tv_diff);
	printf("Performance= %ld.%06ld sec\n", (long int) tv_diff.tv_sec, (long int) tv_diff.tv_usec);

	//Validate for correctness
	for( i = 0; i < XDIM; i++ ){
		if( v[i] != v1[i] + v2[i] ){
			printf("WRONG!!\n");
			exit(EXIT_FAILURE);
		}
	}	
	
	return 0;
}
