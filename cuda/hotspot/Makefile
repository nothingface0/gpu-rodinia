include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

all: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -Xptxas -v -arch=sm_35 -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

test:
	./hotspot 256 2 2 /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/hotspot/temp_256 /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/hotspot/power_256 golden.txt

golden:
	./hotspot 256 2 2 /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/hotspot/temp_256 /home/dsartzet/software/nvbit_release/tools/nvbitfi/logs/input/hotspot/power_256 golden.txt >golden_stdout.txt 2>golden_stderr.txt

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
