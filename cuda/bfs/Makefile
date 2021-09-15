include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs.out

all: $(SRC)
	$(CC) $(SRC) -o $(EXE) -Xptxas -v -arch=sm_35 -I$(INCLUDE) -I../util -L$(CUDA_LIB_DIR)

test:
	./bfs.out /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/bfs/graph32k.txt

golden:
	./bfs.out /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/bfs/graph32k.txt >golden_stdout.txt 2>golden_stderr.txt

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
