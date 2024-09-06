#include<iostream>

__global__ void matrixMul(int *A, int *B, int *C, int NAR, int NAC, int NBR, int NBC) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = tid / NBC, c = tid % NBC;
    int sum = 0;
    for (int i=0; i<NAC; ++i) sum += A[r*NAC + i]*B[i*NBC + c];
    C[r*NAR + c] = sum;
}

__global__ void matrixMul2d(int *A, int *B, int *C, const int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int i=0; i<N; ++i) sum += A[row*N + i]*B[i*N + col];
        C[row*N + col] = sum;
    }
}

// Shared memory size (16 * 16)
#define BLOCK_DIM 16

__global__ void matrixMulTiled(int *A, int *B, int *C, const int N) {
    // Create shared memory for caching per thread block
    __shared__ int sh_a[BLOCK_DIM * BLOCK_DIM];
    __shared__ int sh_b[BLOCK_DIM * BLOCK_DIM];

    const auto ty = threadIdx.y, tx = threadIdx.x;
    const auto by = blockIdx.y, bx = blockIdx.x;
    
    // Compute the global row/col indices
    const auto row = by * BLOCK_DIM + ty, col = bx * BLOCK_DIM + tx;

    int sum = 0;
    for (int block=0; block<(N/BLOCK_DIM); ++block) {
        // Load the corresponding element into the shmem
        sh_a[ty * BLOCK_DIM + tx] = A[row * N + block * BLOCK_DIM + tx];
        sh_b[ty * BLOCK_DIM + tx] = B[block * BLOCK_DIM * N + ty * N + col];

        // Sync all the threads in the blocks
        __syncthreads();

        // compute the sum from this block
        for (int j=0; j<BLOCK_DIM; ++j) {
            sum += sh_a[ty * BLOCK_DIM + j] * sh_b[BLOCK_DIM * j + tx];
        }

        // Sync all the threads in the blocks
        __syncthreads();
    }
    C[row * N + col] = sum; 
}

int main() {
    // initiliaze the size
    const int N = 1 << 10;
    const int bytes = N*N*sizeof(int);
    
    // Allocate managed memory
    int *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // initialize the matrices
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            int i = r*N + c;
            A[i] = B[i] = 1;
        }
    }

    // Launch the kernel
    // const int NUM_THREADS = N, NUM_BLOCKS = (N*N + NUM_THREADS -1) / NUM_THREADS;
    // matrixMul<<<NUM_BLOCKS, NUM_THREADS>>>(A, B, C, N, N, N, N);

    // Launch the 2D kernel
    // const int THREADS_PER_DIM = 16, BLOCKS_PER_DIM = (N + THREADS_PER_DIM -1) / THREADS_PER_DIM;
    // dim3 blocks(BLOCKS_PER_DIM, BLOCKS_PER_DIM);
    // dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM); 
    // matrixMul2d<<<blocks, threads>>>(A, B, C, N);

    // Launch the 2D kernel tiled version
    const int THREADS_PER_DIM = BLOCK_DIM, BLOCKS_PER_DIM = (N + THREADS_PER_DIM -1) / THREADS_PER_DIM;
    dim3 blocks(BLOCKS_PER_DIM, BLOCKS_PER_DIM);
    dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    matrixMulTiled<<<blocks, threads>>>(A, B, C, N);

    // Synchronize with GPU
    cudaDeviceSynchronize();

    // Validate the matrix
    int maxError = 0;
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            maxError = max(maxError, abs(C[r*N + c] - N));
        }
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Free the managed memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}