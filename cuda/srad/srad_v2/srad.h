#define STR_SIZE 256

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#define GPU
// #define TIMER
#define COMPARE // Compare values in results.txt with calculated results.
// #define OUTPUT  // Print output to stdout, so that you can store it in results.txt
