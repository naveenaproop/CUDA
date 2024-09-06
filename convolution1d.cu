#include <iostream>
#include <chrono>

__global__ void convolution1d(const int *A, const int n, const int *mask, const int m, int *C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int radius = m/2; // assuming m is odd
    int c = 0;
    for (int i=max(0, idx-radius); i<=min(n, idx+radius); ++i) {
        c += A[i] * mask[i-idx+radius];
    }
    C[idx] = c;
}

#define MASK_LENGTH 7
__constant__ int cmask[MASK_LENGTH];

__global__ void convolution1dConstMem(const int *A, const int n, int *C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int radius = MASK_LENGTH/2; // assuming m is odd
    int c = 0;
    for (int i=max(0, idx-radius); i<=min(n, idx+radius); ++i) {
        c += A[i] * cmask[i-idx+radius];
    }
    C[idx] = c;
}

#define NUM_THREADS 256
__global__ void convolution1dSharedMem(const int *A, const int n, int *C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int radius = MASK_LENGTH/2; // assuming m is odd
    // Shared memory for the block
    const int NUM_SHARED = NUM_THREADS+2*radius;
    __shared__ int sdata[NUM_SHARED];

    // Copy the corresponding element
    sdata[threadIdx.x] = A[idx];
    if (blockDim.x + threadIdx.x < NUM_SHARED) sdata[blockDim.x + threadIdx.x] = A[blockDim.x + idx];
    __syncthreads();

    // Compute the element by convolving
    int c = 0;
    for (int i=0; i<MASK_LENGTH; ++i) {
        c += sdata[threadIdx.x+i] * cmask[i];
    }
    C[idx] = c;
}


void initiliaze(int *A, const int n, int *mask, const int m, const int offset = 0) {
    for (int i=0; i<n; ++i) A[i+offset] = rand() % 100;
    for (int i=0; i<m; ++i) mask[i] = rand() % 10;
}

void validate(const int *A, const int n, const int *mask, const int m, const int *C, const int offset = 0) {
    const auto begin = std::chrono::steady_clock::now();

    const int radius = m/2; // assuming m is odd
    int maxError = 0;
    for (int idx=0; idx<n; ++idx) {
        int c = 0;
        for (int i=max(0, idx-radius); i<=min(n, idx+radius); ++i) {
            c += A[i+offset] * mask[i-idx+radius];
        }
        maxError = max(maxError, abs(C[idx]-c));
    }
    std::cout << "maxError: " << maxError << std::endl;
    
    std::cout << "Elapsed = " << std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now() - begin).count() << " ns" << std::endl;
}

int main() {
    // Declare variables
    int *A, *C, *mask;
    const int n = 1 << 20, m = 7, r = m/2;
    
    // Allocate unified memory
    // cudaMallocManaged(&A, n*sizeof(int));
    cudaMallocManaged(&A, (n+2*r)*sizeof(int));
    cudaMallocManaged(&C, n*sizeof(int));
    cudaMallocManaged(&mask, m*sizeof(int));

    // Initialize data
    initiliaze(A, n, mask, m, r);
    for (int i=0; i<r; ++i) A[i] = A[n+r+i] = 0;

    // Set the threads and blocks
    const int THREADS = 256, BLOCKS = (n + THREADS - 1) / THREADS;
    
    // Launch Kernel
    // convolution1d<<<BLOCKS, THREADS>>>(A, n, mask, m, C);
    cudaMemcpyToSymbol(cmask, mask, MASK_LENGTH*sizeof(int));
    // convolution1dConstMem<<<BLOCKS, THREADS>>>(A, n, C);
    convolution1dSharedMem<<<BLOCKS, THREADS>>>(A, n, C);

    // Wait for GPU to finish before accessing memory on host
    cudaDeviceSynchronize();

    // Validate
    validate(A, n, mask, m, C, r);

    // Free memory
    cudaFree(mask);
    cudaFree(C);
    cudaFree(A);

    return 0;
}