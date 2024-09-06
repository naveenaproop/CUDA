#include <iostream>
#include <chrono>

__global__ void square(const int *A, const int N, int *C) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int a = A[idx];
        C[idx] = a * a;
    }
}

void initialize(int *A, const int N) {
    for (int i=0; i<N; ++i) A[i] = rand() % 1000;
}

void validate(const int *A, const int N, const int *C) {
    const auto begin = std::chrono::steady_clock::now();
    int maxError = 0;
    for (int i=0; i<N; ++i) {
        maxError = max(maxError, abs(C[i] - A[i] * A[i]));
    }

    std::cout << "maxError: " << maxError << std::endl;
    
    std::cout << "Elapsed = " << std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now() - begin).count() << " ns" << std::endl;
} 
int main() {
    int *A, *C;
    const int N = 1 << 24;

    // Allocate unified memory
    cudaMallocManaged(&A, N*sizeof(int));
    cudaMallocManaged(&C, N*sizeof(int));

    // init with data
    initialize(A, N);

    // Lauch kernel
    const int THREADS = 256, BLOCKS = (N + THREADS - 1) / THREADS;
    square<<<BLOCKS, THREADS>>>(A, N, C);

    // Wait for all GPU threads to finish
    cudaDeviceSynchronize();

    // validate
    validate(A, N, C);

    // Free Memory
    cudaFree(A);
    cudaFree(C);

    return 0;
}