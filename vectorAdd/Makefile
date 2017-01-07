NVCC = nvcc
GCC = gcc

all: vector_add_async vector_add_sync

vector_add_async.o: vectorAdd.cu
	$(NVCC) -o $@ -c $<

vector_add_async: vector_add_async.o
	$(NVCC) -o $@ $+

vector_add_sync.o: vectorAdd.c
	$(GCC) -o $@ -c $<

vector_add_sync: vector_add_sync.o
	$(GCC) -o $@ $+

run:
	./vector_add_sync
	./vector_add_async

diff: 
	diff out.txt parallelout.txt

clean: 
	rm -f *.o *.txt vector_add_sync vector_add_async
