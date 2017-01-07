NVCC = nvcc
GCC = gcc

all: is_prime_sync is_prime_async

is_prime_sync.o: is_prime.c
	$(GCC) -o $@ -c $<

is_prime_sync: is_prime_sync.o
	$(GCC) -o $@ $+
	@rm -f *.o

is_prime_async.o: is_prime.cu
	$(NVCC) -o $@ -c $< -arch=sm_30

is_prime_async: is_prime_async.o
	$(NVCC) -o $@ $+
	@rm -f *.o
	
run:
	@./is_prime_sync < input.txt 
	@./is_prime_async < input.txt

clean:
	@rm -f -r *.o is_prime_sync is_prime_async *.out *.dSYM
