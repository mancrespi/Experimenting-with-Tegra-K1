--------------------------------------------------------------------------------------------------------------
Experimenting with the Tegra K1 on the Jetson TK1
--------------------------------------------------------------------------------------------------------------
Emanuelle Crespi - Class of 2016
--------------------------------------------------------------------------------------------------------------
University of Maryland College Park
Department of Electrical and Computer Engineering 
James A. Clark's School of Engineering 

The following directories include both .c and .cu files that may be executed on NVIDIA's Embedded Jetson TK1 board running CUDA version 6.5. The executables 'genFile50MG' and 'genFile100MG' may be used to generate file.txt at a size of 50-100MG.

- vectorAdd
	a simple vector addition to compare the performance of serial and parallel execution
	serial ver: vectorAdd.c
	CUDA ver:	vectorAdd.cu

- array_file_search
 	implements an array search to read (50-100) bytes from 'file.txt' and search for a specified element within that vectored data
 	serial ver: array_search.c
 	CUDA ver:	array_search.cu

- hash
	implements a one way hash on some vectored data from a set of size 2n --> n
	serial ver: hash.c
	CUDA ver:	hash.cu

- is_prime
	implements an algorithm to test a significantly large integer for prime
	serial ver: is_prime.c
	CUDA ver:	is_prime.cu

- encrypt
	implements a simple xor encryption using 'key.txt'(1MB) to obfuscate 'file.txt'(50-100MB) (incomplete)
	serial ver: enc2.c
	CUDA ver:	enc2.cu

--------------------------------------------------------------------------------------------------------------