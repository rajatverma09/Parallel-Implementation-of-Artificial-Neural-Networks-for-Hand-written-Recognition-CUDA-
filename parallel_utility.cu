#include "cuda_utility.cu"
#define threadsPerBlock 32


#define IFOR(v, s, e) for(int v = s; v < e; ++v)
#define UFOR(v, s, e) for(unsigned v = s; v < e; v++)
#define IFORS(v, s, e, step) for(int v = s; v < e; v += step)

using namespace std;

class ParallelUtility
{
public:
    void init_2D_mat(double **(&arr), int row, int col) {
        arr = new (double*) [row * sizeof(double *)];
        IFOR(i, 0, row)
            arr[i] = new double [col * sizeof(double)];
    }
    
    double* serialize_2D_mat(double **mat,
                             int r,
                             int c)
    {
        int k = 0;
        double *result = new double[r*c];
        IFOR(i, 0, r)
            IFOR(j, 0, c) {
                result[k] = mat[i][j];
                ++k;
            }

        return result;
    }

    double **deserialize_2D_mat(double *arr,
                                int r,
                                int c)
    {
        int k = 0;
        double **res = NULL;
        init_2D_mat(res, r, c);
        IFOR(i, 0, r)
            IFOR(j, 0, c)
                res[i][j] = arr[k++];

        return res;
    }

    void block_and_grid_dim_get(int len, size_t& block_size, size_t& num_blocks) {
        block_size = threadsPerBlock;
        num_blocks = (len / block_size) + ((len % block_size) ? 1 : 0);
    }

    
    double **cuda_mat_transpose_helper(double **hostA, int numARows, int numAColumns)
    {
        double *hostC = (double *) malloc(numCRows * numCColumns * sizeof(double));
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
        
        int numCRows = numAColumns, numCColumns = numARows;
        double *deviceA = NULL, *deviceC = NULL;
        
        cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(double));
        cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(double));
        
        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);
        
        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
        
        cuda_mat_transpose<<<num_blocks, block_size>>>(deviceA, deviceC, numAColumns, numARows, numARows * numAColumns);
        
        cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);
        
        cudaFree(deviceA); cudaFree(deviceC);
        
        return deserialize_2D_mat(hostC, numCRows, numCColumns);
    }
    
    
    double **cuda_mat_multiply_helper(double **hostA, double **hostB,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns)
    {
        int numCRows = numARows, numCColumns = numBColumns;
        double* hostC = (double *) malloc(numCRows * numCColumns * sizeof(double));

        double *devA = NULL, *devB = NULL, *devC = NULL;

        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
        double *hostB_serial = serialize_2D_mat(hostB, numBRows, numBColumns);
        
        cudaMalloc((void**) &devA, numARows * numAColumns * sizeof(double));
        cudaMalloc((void**) &devB, numBRows * numBColumns * sizeof(double));
        cudaMalloc((void**) &devC, numCRows * numCColumns * sizeof(double));
        
        cudaMemcpy(devA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devB, hostB_serial, numBRows * numBColumns * sizeof(double), cudaMemcpyHostToDevice);
        
        dim3 dimGrid(1 + (numCColumns/32), 1 + (numCRows/32), 1);
        dim3 dimBlock(32, 32, 1);    
        
        cuda_mat_multiply <<<dimGrid, dimBlock>>> (devA, devB, devC,
                                                   numARows, numAColumns,
                                                   numBRows, numBColumns,
                                                   numCRows, numCColumns);
        
        cudaMemcpy(hostC, devC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);
        
        cudaFree(devA); cudaFree(devB); cudaFree(devC);
        
        return deserialize_2D_mat(hostC, numCRows, numCColumns);    
    }
    
    
    double **cu_addition_helper(double **hostA, double **hostB,
                                int numARows, int numAColumns)
    {
        double * hostC = (double *) malloc(sizeof(double)*numCRows*numCColumns);

        double *deviceA, *deviceB, *deviceC;
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
        double *hostB_serial = serialize_2D_mat(hostB, numARows, numAColumns);
        
        int numCRows = numARows, numCColumns = numAColumns;
        
        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));
        cudaMalloc((void **)&deviceB, numARows * numAColumns * sizeof(double));
        cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(double));
        
        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);        
        
        cu_addition<<<num_blocks, block_size>>>(deviceA, deviceB, deviceC, numARows * numAColumns);
        
        cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceB); cudaFree(deviceC); cudaFree(deviceA);

        return deserialize_2D_mat(hostC, numCRows, numCColumns);
    }

    
    double** cu_mat_scalar_multiply_helper(double **hostA, double scalar, int numARows, int numAColumns)
    {
        double *deviceA = NULL;
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);

        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));
        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
       
        cu_mat_scalar_multiply<<<num_blocks, block_size>>>(deviceA, scalar, numARows * numAColumns);

        cudaMemcpy(hostA_serial, deviceA, numARows * numAColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA);

        return deserialize_2D_mat(hostA_serial, numARows, numAColumns);
    }


    double** cu_mat_elementwise_multiply_helper(double **hostA, double **hostB,
                                                int numARows, int numAColumns)
    {
        double *deviceA = NULL, *deviceB = NULL;
        double *hostB_serial = serialize_2D_mat(hostB, numARows, numAColumns);
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);

        cudaMalloc((void **)&deviceB, numARows * numAColumns * sizeof(double));
        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));

        cudaMemcpy(deviceB, hostB_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
        
        cu_elementWiseMultiply<<<num_blocks, block_size>>>(deviceA, deviceB, numARows * numAColumns);

        cudaMemcpy(hostA_serial, deviceA, numARows * numAColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceB); cudaFree(deviceA);

        return deserialize_2D_mat(hostA_serial, numARows, numAColumns);
    }

    
    double **cu_sigmoid_helper(double **hostA, int numARows, int numAColumns){
        
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
        double * hostC, * deviceA, * deviceC;

        int numCRows = numAColumns, numCColumns = numARows;
        hostC = (double *) malloc(numCRows * numCColumns * sizeof(double));

        cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(double));
        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));

        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
        
        cu_sigmoid<<<num_blocks, block_size>>>(deviceA, deviceC, numARows * numAColumns);
    
        cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA); cudaFree(deviceC);
        
        return deserialize_2D_mat(hostC, numCRows, numCColumns);
    }


    double** cu_2D_1D_addition_helper(double **hostA, double *hostB,
                                      int numARows, int numAColumns)
    {
        double *hostB_converted = (double*) malloc (numARows * numAColumns * sizeof(double));
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);
        int k = 0;
        IFOR(i, 0, numAColumns) {
            IFOR(j, 0, numARows) {
                hostB_converted[k++] = hostB[i];
            }
        }
        double *hostC = (double *) malloc(numCRows * numCColumns * sizeof(double));
        double *deviceA, *deviceB, *deviceC;

        int numCRows = numARows, numCColumns = numAColumns;

        cudaMalloc((void **)&deviceB, numARows * numAColumns * sizeof(double));
        cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(double));
        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));

        cudaMemcpy(deviceB, hostB_converted, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
        
        cu_addition<<<num_blocks, block_size>>>(deviceA, hostB_converted, deviceC, numARows * numAColumns);
        
        cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);

        return deserialize_2D_mat(hostC, numCRows, numCColumns);
    }

    double *cu_vec_addition_helper(double *hostA, double *hostB, int n){

        double *hostC = (double *) malloc(sizeof(double) * n);
        double *deviceA = NULL, *deviceB = NULL, *deviceC = NULL;

        cudaMalloc((void **)&deviceA, n * sizeof(double));
        cudaMalloc((void **)&deviceB, n * sizeof(double));
        cudaMalloc((void **)&deviceC, n * sizeof(double));

        cudaMemcpy(deviceA, hostA, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB, n * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(n, block_size, num_blocks);
        
        cu_addition<<<num_blocks, block_size>>>(deviceA, deviceB, deviceC, n);

        cudaMemcpy(hostC, deviceC, n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
        return hostC;
    }

    double **cu_dsigmoid_helper(double **hostA, int numARows, int numAColumns)
    {
        double *hostC = (double *) malloc(numCRows * numCColumns * sizeof(double));
        double *deviceA = NULL, *deviceC = NULL;
        double *hostA_serial = serialize_2D_mat(hostA, numARows, numAColumns);

        int numCRows = numAColumns, numCColumns = numARows;

        cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(double));
        cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(double));

        cudaMemcpy(deviceA, hostA_serial, numARows * numAColumns * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(numARows * numAColumns, block_size, num_blocks);
        
        cu_dsigmoid<<<num_blocks, block_size>>>(deviceA, deviceC, numARows * numAColumns);
    
        cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA); cudaFree(deviceC);

        return deserialize_2D_mat(hostC, numCRows, numCColumns);
    }


    double* cu_vec_scalar_multiply_helper(double *hostA, double scalar, int n){
        double * deviceA; 
        cudaMalloc((void **)&deviceA, n * sizeof(double));

        cudaMemcpy(deviceA, hostA, n * sizeof(double), cudaMemcpyHostToDevice);

        size_t block_size, num_blocks;
        block_and_grid_dim_get(n, block_size, num_blocks);
        
        cu_mat_scalar_multiply<<<num_blocks, block_size>>>(deviceA, scalar, n);

        cudaMemcpy(hostA, deviceA, n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(deviceA);         
        return hostA;
    }
};
