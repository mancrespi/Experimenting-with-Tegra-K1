/**
 * Primality Test Serial (Fall 2016): 
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements an algorithm to test for primality discussed in the methodology section
 * of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the function call is_prime(int n, char *factor, char *prime)
 * to perform a sequential search for some factor of the value n.
 *
 * n is purposely chosen as the value n=900000006 for comparison of efficiently determining 
 * a factor in the is_prime.cu code
 *
 *
 * The output of the performance is displayed in seconds for verification.
 *
 */

// System includes
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>


// Performs a sequential search for a factor of the value n
// The algorithm is purposely inefficient to always perform 
// the worst case search of length n.  This is to assist in 
// performing a power analysis during runtime.
//
// When a multiple is found, prime is written to 1 and facter
// is written as the largest multiple 
// Both are meant to be read & verified by the caller
void is_prime(int n, int *factor, int *prime) {
	int i;
	*prime = 1;
	for (i = 2; i < n; i++) {
		if (n % i == 0) {
			*prime = 0;
			*factor = i;
		}
	}
}

int main(void) {
	//r can be modified to produce as much overhead as needed during testing
	int prime, n=900000006, r=1, i, factor;
	struct timeval tv_start, tv_stop, tv_diff; 

	/* Super inefficient primality testing */
	/* with extra computation for runtime */
	gettimeofday(&tv_start, NULL);
	for (i = 0; i < r; i++) {
		is_prime(n, &factor, &prime);
	}
	gettimeofday(&tv_stop, NULL);
	timersub(&tv_stop, &tv_start, &tv_diff);

	printf("Performance= %ld.%06ld sec\n", (long int) tv_diff.tv_sec, (long int) tv_diff.tv_usec);

	//Confirm whether the value n is prime
	if (prime == 1) {
		printf("%d is prime.\n", n);
	} else {
		printf("%d is NOT prime, %d is a factor!\n", n, factor);
	}
	
	return 0;
}
