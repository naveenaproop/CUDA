#include <iostream>

__global__ void vectorAdd(int N, int *a, int *b, int *c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, tcount = blockDim.x * gridDim.x;
    for (int i=tid; i<N; i+=tcount) c[i] = a[i] + b[i];
}

int main() {
    int *a, *b, *c, N = 1 << 20;
    
    // Allocate unified memory - accessible from CPU or GPU
    cudaMallocManaged(&a, N*sizeof(int));
    cudaMallocManaged(&b, N*sizeof(int));
    cudaMallocManaged(&c, N*sizeof(int));

    // initialize a and b arrays
    for (int i=0; i<N; ++i) {
        a[i] = 1, b[i] = 2, c[i] = 0;
    }
    
    // Launch kernel on GPU
    const int NUM_THREADS = 256, NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>> (N, a, b, c);

    // Wait for GPU to finish before accessing memory on host
    cudaDeviceSynchronize();

    // Validate
    int maxError = 0;
    for (int i=0; i<N; ++i) {
        maxError = max(maxError, abs(c[i]-3));
    }
    std::cout << "maxError: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}