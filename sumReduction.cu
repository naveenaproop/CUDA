#include <iostream>

#define THREADS 1024

__global__ void sumReduction(int *a) {
    // Shared memory for elements
    __shared__ int partialSum[THREADS];

    // Copy the element
    partialSum[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // Stride 2^n
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        // Only corresponding threads
        if (threadIdx.x % (2*stride) == 0) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x+stride];
        }
        __syncthreads();
    }

    // thread 0 copies element back to global memory at blockIdx
    if (threadIdx.x == 0) {
        a[blockIdx.x] = partialSum[0];
    }
}

__global__ void sumReductionOpt1(int *a) {
    // Shared memory for elements
    __shared__ int partialSum[THREADS];

    // Copy the element
    partialSum[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // Stride 2^n
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        // Sequential threads and no %
        int index = 2 * stride * threadIdx.x;
        if (index < blockDim.x) {
            partialSum[index] += partialSum[index+stride];
        }
        __syncthreads();
    }

    // thread 0 copies element back to global memory at blockIdx
    if (threadIdx.x == 0) {
        a[blockIdx.x] = partialSum[0];
    }
}

__global__ void sumReductionOpt2(int *a) {
    // Shared memory for elements
    __shared__ int partialSum[THREADS];

    // Copy the element
    partialSum[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();

    // Stride 2^n
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        // Sequential and contiguous threads
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x+stride];
        }
        __syncthreads();
    }

    // thread 0 copies element back to global memory at blockIdx
    if (threadIdx.x == 0) {
        a[blockIdx.x] = partialSum[0];
    }
}

__global__ void sumReductionOpt3(int *a, int *r) {
    // Shared memory for elements
    __shared__ int partialSum[THREADS];

    // Copy the element and do the first iteration (half the blocks)
    const auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    partialSum[threadIdx.x] = a[idx] + a[idx + blockDim.x];
    __syncthreads();

    // Stride 2^n
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        // Sequential and contiguous threads
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // thread 0 copies element back to global memory at blockIdx
    if (threadIdx.x == 0) {
        r[blockIdx.x] = partialSum[0];
    }
}

__device__ void warpReduce(volatile int *shdata, const int tid) {
    for (int s=32; s>0; s>>=1) {
        shdata[tid] += shdata[tid + s];
    }
}

__global__ void sumReductionOpt4(int *a, int *r) {
    // Shared memory for elements
    __shared__ int partialSum[THREADS];

    // Copy the element and do the first iteration (half the blocks)
    const auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    partialSum[threadIdx.x] = a[idx] + a[idx + blockDim.x];
    __syncthreads();

    // Stride 2^n (stop early and call device function instead)
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        // Sequential and contiguous threads
        if (threadIdx.x < stride) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Avoid syncthreads for last few loops, instead use a device function with volatile
    if (threadIdx.x < 32) warpReduce(partialSum, threadIdx.x);

    // thread 0 copies element back to global memory at blockIdx
    if (threadIdx.x == 0) {
        r[blockIdx.x] = partialSum[0];
    }
}

int main() {
    int *a;
    int N = 1 << 20;

    // Create unified-managed memory
    cudaMallocManaged(&a, N*sizeof(int));
    
    // Initialize the data
    for (int i=0; i<N; ++i) a[i] = 1;

    // Lauch the kernels for sum down to 256 and then full sum
    // const int BLOCKS = (N + THREADS - 1) / THREADS;
    // sumReduction<<<BLOCKS, THREADS>>>(a);
    // sumReduction<<<1, THREADS>>>(a);
    // sumReductionOpt1<<<BLOCKS, THREADS>>>(a);
    // sumReductionOpt1<<<1, THREADS>>>(a);
    // sumReductionOpt2<<<BLOCKS, THREADS>>>(a);
    // sumReductionOpt2<<<1, THREADS>>>(a);
    int *r;
    const int BLOCKS = ((N + THREADS - 1) / THREADS) / 2;
    cudaMallocManaged(&r, BLOCKS*sizeof(int));
    // sumReductionOpt3<<<BLOCKS, THREADS>>>(a, r);
    // sumReductionOpt3<<<1, THREADS>>>(r, r);
    sumReductionOpt4<<<BLOCKS, THREADS>>>(a, r);
    sumReductionOpt4<<<1, THREADS>>>(r, r);

    // Synchronize
    cudaDeviceSynchronize();

    // Verify
    std::cout << "maxError: " << (r[0]-N) << std::endl;
    
    // Free memory
    cudaFree(a);

    return 0;
}