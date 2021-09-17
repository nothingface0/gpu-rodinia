include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

all: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -Xptxas -v -arch=sm_35 -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

test:
	./gaussian -v -f /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/gaussian/matrix80.txt

golden:
	./gaussian -v -f /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/gaussian/matrix80.txt >golden_stdout.txt 2>golden_stderr.txt

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

clean:
	rm gaussian
