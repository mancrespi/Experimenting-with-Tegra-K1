/**
 * Array Search Serial (Fall 2016):
 * 
 * Members:
 * Emanuelle Crespi, Tolga Keskinoglu
 *
 * This test implements an array search discussed in the methodology section
 * of Optimizing CPU-GPU Interactions.
 *
 * The following code makes use of the fucntion call search(int n, char *data, char *out, char c)
 * to perform a sequential search for char c amongst a large vector of characters.
 *
 * The result is a one-one mapping of the data array with 1s and 0s in the out array at 
 * corresponding indices where the character has been found.  The output is written to 
 * the file 'array_search_result.txt' for validation with 'parrallel_array_search_result.txt'
 * when running the executable for array_search.cu
 *
 * The output of the performance is displayed in seconds. 
 * The performance results are to be compared with the performance of array_search.cu
 *
 */

//System includes
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

/* 
 * Performs a sequential search for char c in array data both of size n
 * The output 1 or 0 is written in out to indicate 'found' and 'not-found' respectively
 * search( n, ['a','b','d','c','d','e',...], result, 'd') ==> result = ['0','0','1','0','1','0',...]
 */
void search(int n, char *data, char *out, char c){
	int i;
	for (i = 0; i < n; i++){
		if (data[i] == c){
			out[i] = '1';
		}else{ 
			out[i] = '0';
		}	
	}
}

int main(){
	//Initializations 
	FILE *fp_data, *fp_out;
	char *data, *out, c;
	struct timeval tv_start, tv_stop, tv_diff; 
	int s_data = 0, j, i = 0;

	//sys/time.h types
	clock_t start, diff;

	//printf("Computing file size...\n");
	if (!(fp_data = fopen("../../file.txt", "r")))
		perror("failed to read file\n");
	
	while( fscanf(fp_data,"%c",&c) != EOF ){
		s_data++;
	}
	
	/* Allocate necessary space for buffer */
	//printf("Mallocing %d bytes of data on CPU...\n", s_data);
	data = (char *)malloc(sizeof(char)*s_data);
	out = (char *)malloc(sizeof(char)*s_data);

	fseek(fp_data, 0, 0);

	/* Read file into buffer */
	//printf("Reading data into buffer...\n");	
	for( j = 0; fscanf(fp_data,"%c",&data[j]) != EOF; j++ ){}

	/* Time the search algorithm */
	//printf("Executing search on CPU...\n");
	gettimeofday(&tv_start, NULL);
	printf("Running...\n");
	search(s_data, data, out, 'D');

	gettimeofday(&tv_stop, NULL);
	timersub(&tv_stop, &tv_start, &tv_diff);

	printf("Performance= %ld.%06ld sec\n", (long int) tv_diff.tv_sec, (long int) tv_diff.tv_usec);

	//printf("Writing result to array_search_result.txt...\n");
 	fp_out = fopen("array_search_result.txt", "w");

 	for (j = 0; j < s_data; j++){
		if( i == 32 ){
			fprintf(fp_out, "%c\n", out[j]);
			i = 0;	
		}else{
			fprintf(fp_out, "%c", out[j]);
			i++;
		}
	}

	/* Cleanup */
 	free(data); free(out);
 	fclose(fp_data); fclose(fp_out);
	return 0;
}
