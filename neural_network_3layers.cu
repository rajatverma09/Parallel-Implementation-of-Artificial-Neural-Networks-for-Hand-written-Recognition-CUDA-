#include <bits/stdc++.h>
#include <cuda.h>
#include "parallel_utility.cu"
#include "matrix_utility.cu"

#define DATA_DIM 3072
#define TRAIN_SAMPLE 500
#define TEST_SAMPLE 100

#define OUTPUT_LAYER_NODES 1
#define HIDDEN_LAYER_NODES 5

#define EPOCH 200
#define LEARNING_RATE_IP_HIDDEN 1
#define LEARNING_RATE_HIDDEN_OP 1

#define IFOR(v, s, e) for(int v = s; v < e; ++v)
#define UFOR(v, s, e) for(unsigned v = s; v < e; v++)


using namespace std;

ParallelUtility pu = ParallelUtility();
MatrixUtility mu = MatrixUtility();


class Initializer
{
public:
    void load_data(double **(&data), double *labels, int row, int col, const char *filename){
        float ch;
        FILE *fp = fopen(filename, "r");
        if (!fp)
            return;

        fscanf(fp, "%f", &ch);
        double **dataset = NULL;

        mu.init_2D_mat(dataset, row, col);
        
        int i = 0, j = 0;
        for (int ct = 0; ct < (row * col); ct++)
            dataset[i][j++] = ch;
            if (j == col) {
                j = 0;
                i++;
            }
            fscanf(fp, "%f", &ch);
        }
        fclose(fp);
        fp = NULL
        
        IFOR(k, 0, row) {
            labels[k] = dataset[k][col-1];
            for (j = 0; j < col - 1; ++j)
                data[k][j] = dataset[k][j];
        }
    }

    void init_weights(double **(&w),
                      int row,
                      int col)
    {
        IFOR(i, 0, row)
            IFOR(j, 0, col)    
                w[i][j] = ((double)rand() / (double)RAND_MAX);
    }

    void init_biases(double *(&b), int row) {
        IFOR(i, 0, row)    
            b[i] = ((double)rand() / (double)RAND_MAX);
    }
};

class NeuralNetwork
{
public:

    double **dsigmoid(double **a,
                      int r,
                      int c)
    {
        double **one_minus_sigmoid_a = NULL;
        double **sigmoid_a = sigmoid(a, r, c);

        mu.init_2D_mat(one_minus_sigmoid_a, r, c);
        IFOR(i, 0, r)
            IFOR(j, 0, c)
                one_minus_sigmoid_a[i][j] = 1 - sigmoid_a[i][j];

        return pu.cu_mat_elementwise_multiply_helper(sigmoid_a, one_minus_sigmoid_a, r, c);
    }


    double **sigmoid(double **mat, int r, int c) {
        double **s;
        mu.init_2D_mat(s, r, c);
        UFOR(i, 0, r)        
        {
            UFOR(j, 0, c)        
            {
                s[i][j] = 1 / (1 + exp(-mat[i][j]));
            }
            
        }
        return s;
    }

    
    void back_prop(double **(&X), double *(&Y),
                   double **(&W1), double **(&W2),
                   double **(&A1), double **(&A2),
                   double **(&dW1), double **(&dW2),
                   double **(&dA1), double **(&dA2),
                   double *(&db1), double *(&db2))
    {
        double **one;
        mu.init_2D_mat(one, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        
        UFOR(i, 0, TRAIN_SAMPLE)   
            UFOR(j, 0, HIDDEN_LAYER_NODES)   
                one[i][j] = 1;
        
        double **dZ2 = mu.diff_2D_mat_1D_mat(A2, Y, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
        double **dZ2_trans = pu.cuda_mat_transpose_helper(dZ2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
        
        dW2 = pu.cuda_mat_multiply_helper(dZ2, A1, OUTPUT_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        dW2 = pu.cu_mat_scalar_multiply_helper(dW2, 1/TRAIN_SAMPLE, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
        db2 = mu.sum_across_2nd_dim(dZ2_trans, OUTPUT_LAYER_NODES, TRAIN_SAMPLE);
        db2 = pu.cu_vec_scalar_multiply_helper(db2, 1/TRAIN_SAMPLE, OUTPUT_LAYER_NODES);

        double **A1_square = pu.cu_mat_elementwise_multiply_helper(A1, A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        A1_square = pu.cu_mat_scalar_multiply_helper(A1_square, -1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        A1_square = pu.cu_addition_helper(one, A1_square, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        
        
        double **W2xdZ2 = pu.cuda_mat_multiply_helper(dZ2, W2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
        double **derivative_Z1 = dsigmoid(A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);

        double **dZ1 = pu.cu_mat_elementwise_multiply_helper(derivative_Z1, W2xdZ2, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        double **dZ1_trans = pu.cuda_mat_transpose_helper(dZ1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
        dW1 = pu.cuda_mat_multiply_helper(dZ1_trans, X, HIDDEN_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, DATA_DIM);
        dW1 = pu.cu_mat_scalar_multiply_helper(dW1, 1/TRAIN_SAMPLE, HIDDEN_LAYER_NODES, DATA_DIM);
        db1 = mu.sum_across_2nd_dim(dZ1_trans, HIDDEN_LAYER_NODES, TRAIN_SAMPLE);
        db1 = pu.cu_vec_scalar_multiply_helper(db1, 1/TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    }


    void forward_prop(double **(&X), double **(&W1), double **(&W2),
                                     double *(&b1), double *(&b2),
                                     double **(&Z1), double **(&Z2),
                                     double **(&A1), double **(&A2),
                                     int examples)
    {
        double **W1_trans = NULL;
        W1_trans = pu.cuda_mat_transpose_helper(W1, HIDDEN_LAYER_NODES, DATA_DIM);
        
        Z1 = pu.cuda_mat_multiply_helper(X, W1_trans, examples, DATA_DIM, DATA_DIM, HIDDEN_LAYER_NODES);
        A1 = sigmoid(Z1, examples, HIDDEN_LAYER_NODES);
        
        A1 = mu.add_2D_mat_1D_mat(A1, b1, examples, HIDDEN_LAYER_NODES);
        
        double **W2_trans = pu.cuda_mat_transpose_helper(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
        
        Z2 = pu.cuda_mat_multiply_helper(A1, W2_trans, examples, HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES, OUTPUT_LAYER_NODES);
        A2 = sigmoid(Z2, examples, OUTPUT_LAYER_NODES);
        
        A2 = mu.add_2D_mat_1D_mat(A2, b2, examples, OUTPUT_LAYER_NODES);
    }


    void update_parameter(double **(&W1), double **(&W2), double *(&b1), double *(&b2),
                        double **(&dW1), double **(&dW2), double *(&db1), double *(&db2))
    {
        dW2 = pu.cu_mat_scalar_multiply_helper(dW2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
        dW1 = pu.cu_mat_scalar_multiply_helper(dW1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES, DATA_DIM);
        
        W1 = pu.cu_addition_helper(W1, dW1, HIDDEN_LAYER_NODES, DATA_DIM);
        W2 = pu.cu_addition_helper (W2, dW2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
        
        db1 = pu.cu_vec_scalar_multiply_helper(db1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES);
        db2 = pu.cu_vec_scalar_multiply_helper(db2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES);
        
        b1 = pu.cu_vec_addition_helper(b1, db1, HIDDEN_LAYER_NODES);
        b2 = pu.cu_vec_addition_helper(b2, db2, OUTPUT_LAYER_NODES);
    }

};
