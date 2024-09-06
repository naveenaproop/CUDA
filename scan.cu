#include <iostream>
#include <chrono>

__global__ void scan(const int *A, const int N, int *C) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Copy the elements into C first
    C[idx] = A[idx];
    __syncthreads();

    for (int s=1; idx >= s && s<N; s<<=1) {
        int tmp = C[idx-s];
        __syncthreads();
        C[idx] += tmp;
        __syncthreads();
    }
}

void initialize(int *A, const int N) {
    for (int i=0; i<N; ++i) A[i] = 1;// rand() % 100;
}

void validate(const int *A, const int N, const int *C) {
    const auto begin = std::chrono::steady_clock::now();

    int maxError = 0, cum = 0;
    for (int i=0; i<N; ++i) {
        cum += A[i];
        maxError = max(maxError, abs(C[i] - cum));
        // std::cout << A[i] << "\t" << C[i] << "\t" << cum << "\t" << maxError << std::endl;
    }
    std::cout << "CPU Elapsed: " << std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - begin).count() << " us" << std::endl;
    std::cout << "maxError: " << maxError << std::endl;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int *A, *C;
    const int N = 1 << 10;
    std::cout << "N " << N << std::endl;

    cudaMallocManaged(&A, N*sizeof(int));
    cudaMallocManaged(&C, N*sizeof(int));

    initialize(A, N);

    const int T = 1024, B = (N + T - 1) / T;
    cudaEventRecord(start);
    scan<<<B, T>>>(A, N, C);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    validate(A, N, C);

    cudaFree(C);
    cudaFree(A);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Elapsed: " << int(milliseconds*1000) << " us" << std::endl;
    
    return 0;
}