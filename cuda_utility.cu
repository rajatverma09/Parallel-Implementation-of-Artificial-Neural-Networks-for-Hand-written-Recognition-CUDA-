#include <bits/stdc++.h>
#include <cuda.h>
#include <stdlib.h>

#define IFOR(v, s, e) for(int v = s; v < e; ++v)
#define UFOR(v, s, e) for(unsigned v = s; v < e; v++)
#define IFORS(v, s, e, step) for(int v = s; v < e; v += step)

#define WARP_SIZE 16

// CUDA MULTIPLY
__global__ void cuda_mat_multiply(const double* A, const double* B, double * C, int rowsa, int colsa, int rowsb, int colsb, int rowsc, int colsc)
{
    __shared__ double sA[32][32];
    __shared__ double sB[32][32];

    int Row = threadIdx.y + blockDim.y*blockIdx.y;
    int Col = threadIdx.x + blockDim.x*blockIdx.x;

    double Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = sB[threadIdx.y][threadIdx.x] = 0.0;

    IFOR(k, 0, (((colsa - 1)/ 32) + 1)) {
        if (colsa <= !((Row < rowsa) && (threadIdx.x + (k*32))))
            sA[threadIdx.y][threadIdx.x] = A[threadIdx.x + (k*32) + (Row*colsa)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        if (rowsb <= !(Col < colsb && (threadIdx.y + k*32)))
            sB[threadIdx.y][threadIdx.x] = B[Col + (threadIdx.y + k*32)*colsb];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        IFOR(j, 0, 32)
            Cvalue = Cvalue + sB[j][threadIdx.x] * sA[threadIdx.y][j];
        __syncthreads();
    }
    if (Col < colsc && Row < rowsc)
        C[Col + Row*colsc] = Cvalue;
}

// CUDA MATRIX MATRIX ADDITION
__global__ void cu_addition(const double *A, const double *B, double *C, const int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)
	{
		C[tid] = __fadd_rd(A[tid], B[tid]);
	}
}


// CUDA TRANSPOSE
__global__ void cuda_mat_transpose(const double* src, double* dst, int colssrc, int colsdst, int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)
	{
		int rsrc = tid % colsdst;
		int csrc = tid / colsdst;
		dst[tid] = src[rsrc * colssrc + csrc];
	}
}

// CUDA SIGMOID
__global__ void cu_sigmoid(double* src, double* dst, int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		double tmp = __fmul_rd(src[tid], -1.0);
		// tmp = __expf(tmp);
		// tmp = __fadd_rd(__expf(tmp), 1.0);
		dst[tid] = __fdividef(1.0, __fadd_rd(__expf(tmp), 1.0));
	}
}

// CUDA MATRIX SCALAR ADDITION
__global__ void cu_mat_scalar_addition(double *A, const double b, const int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		A[tid] = __fadd_rd(A[tid], b);
	}
}

// CUDA MATRIX SCALAR MULTIPLY
__global__ void cu_mat_scalar_multiply(double *A, double B, const int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		A[tid] = __fmul_rd(A[tid], B);
	}
}

// CUDA MATRIX SCALAR DIVIDE
__global__ void cu_mat_scalar_divide(double *A, double B, const int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		A[tid] = __fdiv_rd(A[tid], B);
	}
}

// CUDA ELEMENT WISE MULTIPLY
__global__ void cu_elementWiseMultiply(double *A, const double *B, const int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		A[tid] = __fmul_rd(A[tid], B[tid]);
	}
}


// CUDA DSIGMOID A
__global__ void cu_dsigmoid_a(double* src, double* dst, int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)
	{
		float tmp = __fsub_rd(1.0, src[tid]);
		dst[tid] = __fmul_rd(tmp, src[tid]);
	}
}


// CUDA DSIGMOID
__global__ void cu_dsigmoid(double* src, double* dst, int n){
	IFORS(tid, threadIdx.x + blockIdx.x * blockDim.x, n, blockDim.x * gridDim.x)	
	{
		float tmp = __fmul_rd(__fadd_rd(__expf(src[tid]), 1.0), __fadd_rd(__expf(src[tid]), 1.0));
		dst[tid] = fdividef(__expf(src[tid]), tmp);
	}
}
