CC = aarch64-linux-gnu-g++
NVCC = /usr/local/cuda-10.0/bin/nvcc -ccbin $(CC)
NVCC_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib
INCLUDES = /usr/local/cuda-10.0/samples/common/inc
CUDART_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so

CXXFLAGS = -std=c++0x -fopenmp

Project2: project2.o
	$(CC) $(CXXFLAGS) -o Project2 project2.o -L$(NVCC_LIB) -lcudart

project2.o: project2.cu
	$(NVCC) -I. -I$(INCLUDES) -c project2.cu

clean:
	rm -rf Project2 project2.o
