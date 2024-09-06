#include <iostream>
#include <cublas_v2.h>

using namespace std;

int main() {
    float *a, *b;
    const int N = 1 << 20, bytes = N*sizeof(int);
    
    // Allocate unified memory - accessible from CPU or GPU
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);

    // initialize a and b arrays
    for (int i=0; i<N; ++i) {
        a[i] = 1, b[i] = 2; //, c[i] = 0;
    }
    
    // Create CUBLAS handle and copy vectors
    cublasHandle_t blas;
    auto status = cublasCreate_v2(&blas);

    // CUBLAS for vector addition, c = alpha * a + y
    float alpha = 1.0f;
    cublasSaxpy(blas, N, &alpha, a, 1, b, 1);

    // Wait for GPU to finish before accessing memory on host
    cudaDeviceSynchronize();

    cublasDestroy_v2(blas);

    // Validate
    float maxError = 0;
    for (int i=0; i<N; ++i) {
        // maxError = max(maxError, abs(c[i]-3));
        maxError = max(maxError, abs(b[i]-3));
    }
    std::cout << "maxError: " << maxError << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);

    return 0;
}