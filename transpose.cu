#include <iostream>
#include <chrono>
#include "gputimer.h"

const int N = 1 << 10;
const int K = 16;

__global__ void transpose_serial(const int *A, int *C) {
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            C[r*N + c] = A[r + c*N];
        }
    }
}

__global__ void transpose_per_row(const int *A, int *C) {
    int r = threadIdx.x;
    for (int c=0; c<N; ++c) {
        C[r*N + c] = A[r + c*N];
    }
}

__global__ void transpose_per_elem1(const int *A, int *C) {
    const int r = blockIdx.x, c = threadIdx.x;
    C[r*N + c] = A[r + c*N];
}

__global__ void transpose_per_elem2(const int *A, int *C) {
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    // C[r*N + c] = A[r + c*N];
    C[r + c*N] = A[r*N + c];
}

__global__ void transpose_per_elem_tiling(const int *A, int *C) {
    __shared__ int tile[K][K];
    
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const int r = by * K + ty, c = bx * K + tx;
    tile[tx][ty] = A[r*N + c];

    __syncthreads();

    int nr = bx * K + ty, nc = by * K + tx;
    C[nr*N + nc] = tile[ty][tx];
}

void initialize(int *A) {
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            A[r*N + c] = rand() % 100;
        }
    }
}

void validate(const int *A, const int *C) {
    const auto begin = std::chrono::steady_clock::now();

    int maxError = 0;
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            maxError = max(maxError, abs(C[r*N + c] - A[r + c*N]));
        }
    }
    
    std::cout << "CPU Elapsed: " << std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::steady_clock::now() - begin).count() << " us" << std::endl;
    std::cout << "maxError: " << maxError << (maxError == 0 ? " - SUCCESS!" : " - FAILED!") << std::endl;
}

int main() {
    int *A, *C;

    cudaMallocManaged(&A, N*N*sizeof(int));
    cudaMallocManaged(&C, N*N*sizeof(int));

    initialize(A);

    GpuTimer gt;
    gt.Start();

    // transpose_serial<<<1,1>>>(A, C);
    // transpose_per_row<<<1,N>>>(A, C);
    // transpose_per_elem1<<<N,N>>>(A, C);
    
    dim3 T(K, K), B((N+K-1)/K, (N+K-1)/K);
    // transpose_per_elem2<<<B,T>>>(A, C);

    transpose_per_elem_tiling<<<B,T>>>(A, C);

    cudaDeviceSynchronize();

    gt.Stop();
    std::cout << "GPU Elapsed: " << int(gt.Elapsed()*1000) << " us" << std::endl;

    validate(A, C);

    cudaFree(C);
    cudaFree(A);
    return 0;
}