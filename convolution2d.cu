#include <iostream>
#include <chrono>

#define MASK_DIM 7

__constant__ int cmask[MASK_DIM * MASK_DIM];

__global__ void convolution2d(const int *A, const int n, int *C) {
    // Get the row and col
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Compute the start row/col
    const int startRow = row - MASK_DIM / 2;
    const int startCol = col - MASK_DIM / 2;

    // Convolve and set the value
    int val = 0;
    for (auto r=startRow, i=0; i<MASK_DIM; ++r, ++i) {
        for (auto c=startCol, j=0; j<MASK_DIM; ++c, ++j) {
            if (r >= 0 && r < n && c >= 0 && c < n) {
                val += A[r*n + c] * cmask[i*MASK_DIM + j];
            }
        }
    }
    C[row*n + col] = val;
}

void initialize(int *A, int n, int *mask, int m) {
    for (int r=0; r<n; ++r) {
        for (int c=0; c<n; ++c) {
            A[r*n + c] = rand() % 100;
        }
    }
    for (int r=0; r<m; ++r) {
        for (int c=0; c<m; ++c) {
            mask[r*m + c] = rand() % 10;
        }
    }
}

void validate(const int *A, const int n, const int *mask, const int m, const int *C) {
    const auto begin = std::chrono::steady_clock::now();
    int maxError = 0;
    for (int row=0; row<n; ++row) {
        for (int col=0; col<n; ++col) {
            const int startRow = row - m / 2;
            const int startCol = col - m / 2;
            int val = 0;
            for (auto r=startRow, i=0; i<m; ++r, ++i) {
                for (auto c=startCol, j=0; j<m; ++c, ++j) {
                    if (r >= 0 && r < n && c >= 0 && c < n) {
                        val += A[r*n + c] * mask[i*m + j];
                    }
                }
            }
            maxError = max(maxError, abs(C[row*n + col] - val));
        }
    }
    std::cout << "maxError: " << maxError << std::endl;
    
    std::cout << "Elapsed = " << std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now() - begin).count() << " ns" << std::endl;
}

int main() {
    // Declare variable
    int *A, *mask, *C;
    const int N = 1 << 10;

    // Create unified memory
    cudaMallocManaged(&A, N*N*sizeof(int));
    cudaMallocManaged(&C, N*N*sizeof(int));
    cudaMallocManaged(&mask, MASK_DIM*MASK_DIM*sizeof(int));

    // initialize
    initialize(A, N, mask, MASK_DIM);
    cudaMemcpyToSymbol(cmask, mask, MASK_DIM*MASK_DIM*sizeof(int));

    // Launch kernel
    const int THREADS = 16, BLOCKS = (N + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS), blocks(BLOCKS, BLOCKS);
    convolution2d<<<blocks, threads>>>(A, N, C);

    // Synchronize with GPU
    cudaDeviceSynchronize();

    // validate
    validate(A, N, mask, MASK_DIM, C);

    return 0;
}