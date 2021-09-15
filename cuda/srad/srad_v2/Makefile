include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -DOUTPUT

INCLUDE := $(CUDA_DIR)/include

SRC = srad.cu

EXE = srad

all: clean srad_v2

srad_v2: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -arch=sm_35 -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

test:
	./srad 256 256 0 127 0 127 0.5 2

golden:
	./srad 256 256 0 127 0 127 0.5 2 >golden_stdout.txt 2>golden_stderr.txt

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
