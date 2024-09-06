#include <iostream>
#include <chrono>

#define BINS 7
#define DIV (26 + BINS - 1) / BINS

__global__ void histogram(const char *A, const int N, unsigned int *hist) {
    // Compute the global idx
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        // Increment the bin
        atomicAdd(&hist[(A[idx]-'a') / DIV], 1);
    }
}

__global__ void histogramShMem(const char *A, const int N, unsigned int *hist) {
    // Compute the global idx
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Create Shared memory for this thread block and initialize to 0
    __shared__ unsigned int sh_hist[BINS];
    if (threadIdx.x < BINS) sh_hist[threadIdx.x] = 0;
    __syncthreads();

    for (int i=idx; i < N; i += (gridDim.x * blockDim.x)) {
        // Increment the bin
        atomicAdd(&sh_hist[(A[i]-'a') / DIV], 1);
    }
    __syncthreads();

    // Add the local bins to global bins
    if (threadIdx.x < BINS) {
        atomicAdd(&hist[threadIdx.x], sh_hist[threadIdx.x]);
    }
}

void initialize(char *A, const int N, unsigned int *hist) {
    for (int i=0; i<N; ++i) A[i] = 'a' + rand() % 26;
    for (int i=0; i<BINS; ++i) hist[i] = 0;
}

void validate(const char *A, const int N, const unsigned int *hist) {
    const auto begin = std::chrono::steady_clock::now();
    unsigned int freq[BINS]{0};

    for (int i=0; i<N; ++i) ++freq[(A[i]-'a')/DIV];

    int maxError = 0;
    for (int i=0; i<BINS; ++i) maxError = max(maxError, abs(int(hist[i])-int(freq[i])));

    std::cout << "maxError: " << maxError << std::endl;
    
    std::cout << "Elapsed = " << std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now() - begin).count() << " ns" << std::endl;
}

int main() {
    char *A;
    unsigned int *hist;
    const int N = 1 << 20;

    // Allocate unified memory
    cudaMallocManaged(&A, N*sizeof(int));
    cudaMallocManaged(&hist, BINS*sizeof(unsigned int));

    // initialize
    initialize(A, N, hist);
    
    // Launch kernel
    const int THREADS = 256, BLOCKS = (N + THREADS - 1) / THREADS;
    // histogram<<<BLOCKS, THREADS>>>(A, N, hist);
    histogramShMem<<<BLOCKS / 16, THREADS>>>(A, N, hist);

    // Sync with GPU
    cudaDeviceSynchronize();

    // validate
    validate(A, N, hist);

    // Free
    cudaFree(A);
    cudaFree(hist);

    return 0;
}