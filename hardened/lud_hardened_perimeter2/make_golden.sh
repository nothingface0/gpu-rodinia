make clean
make 2> stderr.txt
cp cuda/lud_cuda .
./lud_cuda -s 128 -v >golden_stdout.txt 2>golden_stderr.txt

