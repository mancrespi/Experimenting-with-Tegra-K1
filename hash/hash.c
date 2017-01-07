/**
 * One Way Hash Test Serial (Fall 2016):
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements a simple hash from a space of size 2n --> n 
 * discussed in the methodology section of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the kernel call hash(char *f, char *h, int n)
 * to perform a sequential hash of elements f --> h with corresponding indices 2i --> i 
 *
 * The result is a mapping of the data within f to the data within h 
 * The output is verified before the program terminates to see that every 
 * element at index 2i of f is indeed at index i in h
 *
 * The output of the performance is displayed in seconds. 
 * The performance results are to be compared with the performance of hash.cu
 *
 */

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//this hash requires even length arrays
#define EVEN_NUM 123374234


// this function performs a hash of n elements into an array h of size n/2
// h is considered a bucket holding all of the elements at even indices in f
void hash(char *f, char *h, int n) {
	int i;

	for( i = 0; i < n; i++){
		h[i] = f[2*i];	
	}
}

int main(void) {
	//r can be modified to produce as much overhead as needed during testing
	int two_n = EVEN_NUM, i, r = 50;
	struct timeval tv_start, tv_stop, tv_diff;
	char *f, *h;

	if ( two_n % 2 ){
		//printf("NO NO NO!!! Even numbers only please.\n");
		exit(EXIT_FAILURE);
	}

	//set up space for arrays
	f = calloc(sizeof(char), two_n + 1);
	h = calloc(sizeof(char), two_n/2 + 1);

	//Populate data into array
	for (i = 0; i < two_n; i++) {
		f[i] = (char) ((i % 94) + 33);
	}

	//Run test
	printf("Running...\n");	
	gettimeofday(&tv_start, NULL);
	for (i = 0; i < r; i++) {
		hash(f, h, two_n/2);
	}
	gettimeofday(&tv_stop, NULL);
	timersub(&tv_stop, &tv_start, &tv_diff);
	printf("Performance= %ld.%06ld sec\n", (long int) tv_diff.tv_sec, (long int) tv_diff.tv_usec);

	//Validate for correctness (takes extra time but avoids overhead from file writing)
	for (i = 0; i < two_n/2; i++) {
		if (h[i] != f[2*i]) {
			printf("WRONG!\n");
			return 1;
		}
	}
	
	return 0;
}
