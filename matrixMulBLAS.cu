#include<iostream>
#include<cublas_v2.h>

int main() {
    // initiliaze the size
    const int N = 1 << 10, bytes = N*N*sizeof(int);
    
    // Allocate managed memory
    float *A, *B, *C;
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

    // Create CUBLAS handle
    cublasHandle_t blas;
    cublasCreate_v2(&blas);

    // Multiply C = (alpha * A) * B + (beta * C)
    float alpha = 1, beta = 0;
    cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);

    // Synchronize with GPU
    cudaDeviceSynchronize();

    // Destroy
    cublasDestroy_v2(blas);

    // Validate the matrix
    float maxError = 0;
    for (int r=0; r<N; ++r) {
        for (int c=0; c<N; ++c) {
            maxError = max(maxError, abs(C[r*N + c] - N));
        }
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Free the managed memory
    for (int i=0; i<N; ++i) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }
    
    return 0;
}