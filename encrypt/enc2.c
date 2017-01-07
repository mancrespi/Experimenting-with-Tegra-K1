#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h> 

void encrypt1(int n, char *m, char *k, char *c){
	int i,j;
	for (i = 0; i < n; i++){
		for( j = 1; j <= 100; j++){
			c[i] = m[i] ^ k[i];
		}		
	}
}

int main(){
	FILE *fp_m, *fp_k;
	char *m, *k, *c, ch;
	struct timeval tv_start, tv_stop, tv_diff;
	int s_m=0, s_k=0, j;

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

	//malloc space for m, k, c
	m = (char *)malloc(sizeof(char)*s_m);
	k = (char *)malloc(sizeof(char)*s_k);
	c = (char *)malloc(sizeof(char)*s_m);

	fseek(fp_m, 0, 0);
	fseek(fp_k, 0, 0);

	//read into buffers
	for( j = 0; fscanf(fp_m,"%c",&m[j]) != EOF; j++ ){}
	for( j = 0; fscanf(fp_k,"%c",&k[j]) != EOF; j++ ){}
	
	gettimeofday(&tv_start, NULL);
 	encrypt1(s_m, m, k, c);
	gettimeofday(&tv_stop, NULL);
	timersub(&tv_stop, &tv_start, &tv_diff);
	printf("Performance= %ld.%06ld sec\n", (long int) tv_diff.tv_sec, (long int) tv_diff.tv_usec);

 	//fp_c = fopen("serial_cipher.txt", "w");
	
	//validate for correctness
 	for (j = 0; j < s_m; j++){
		if( c[j] != (m[j]^k[j]) ){
			printf("WRONG!\n");
			exit(EXIT_FAILURE);
		}
	}

	/* Cleanup */
 	free(m); free(k); free(c);
 	fclose(fp_m); fclose(fp_k);

	return 0;
}
