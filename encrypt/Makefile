NVCC = nvcc
GCC = gcc

all: enc_sync enc_async

enc_sync.o: enc2.c
	$(GCC) -o $@ -c $<

is_prime_sync: enc_sync.o
	$(GCC) -o $@ $+
	@rm -f *.o

enc_async.o: enc2.cu
	$(NVCC) -o $@ -c $< -arch=sm_30

enc_async: enc_async.o
	$(NVCC) -o $@ $+
	@rm -f *.o
	
run:
	@./enc_sync 
	@./enc_async

clean:
	@rm -f -r *.o enc_sync enc_async
