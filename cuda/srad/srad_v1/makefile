include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

# link objects(binaries) together
a.out:		main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcudart -arch sm_10 -o srad

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	nvcc	main.cu \
				-c -O3  -arch sm_10

# delete all object files
clean:
	rm *.o srad
