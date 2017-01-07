NVCC = nvcc
GCC = gcc

all: hash_sync hash_async 

hash_async.o: hash.cu 
	$(NVCC) -o $@ -c $<

hash_async: hash_async.o
	$(NVCC) -o $@ -arch=sm_30 $+

hash_sync.o: hash.c
	$(GCC) -o $@ -c $<

hash_sync: hash_sync.o
	$(GCC) -o $@ $+

run:
	./hash_sync
	./hash_async

clean: 
	rm -f *.o *.txt hash_sync hash_async
