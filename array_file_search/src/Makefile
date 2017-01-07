NVCC = nvcc
GCC = gcc

all: array_search_sync array_search_async 

array_search_async.o: array_search.cu 
	$(NVCC) -o $@ -c $<

array_search_async: array_search_async.o
	$(NVCC) -o $@ -arch=sm_30 $+
#-arch=sm_30
array_search_sync.o: array_search.c
	$(GCC) -o $@ -c $<

array_search_sync: array_search_sync.o
	$(GCC) -o $@ $+

run:
	./array_search_sync
	./array_search_async
	
diff: 
	diff array_search_result.txt parallel_array_search_result.txt

clean: 
	rm -f *.o *.txt array_search_sync array_search_async 
