#include <bits/stdc++.h>
#include <cuda.h>
#include <stdlib.h>

#define IFOR(v, s, e) for(int v = s; v < e; ++v)
#define UFOR(v, s, e) for(unsigned v = s; v < e; v++)

using namespace std;

class MatrixUtility
{
    public:
        void print1Dmat(double *arr, int m) {

            IFOR(i, 0, m)    
                cout << arr[i] << " ";
            cout << '\n';
        }

        void print2Dmat(double **arr, int m, int n) {
            IFOR(i, 0, m)
            {
                IFOR(j, 0, n)
                    printf("%0.9f ", arr[i][j]);
                cout << '\n';
            }
            cout << '\n';
        }

        void init_1D_mat(double *(&arr), int n) {
            arr = (double *)malloc(n * sizeof(double));
        }

        void init_2D_mat(double **(&arr), int row, int col) {
            arr = (double **)malloc(row * sizeof(double *));
            IFOR(i, 0, row)
                arr[i] = (double *)malloc(col * sizeof(double));
        }

        double **mat_add(double **(&A), double **(&B), int row, int col)
        {
            double **res = NULL;
            init_2D_mat(res, row, col);

            IFOR(i, 0, row)
                IFOR(j, 0, col)
                res[i][j] = A[i][j] + B[i][j];
            return res;
        }

        double **mat_multiply(double **(&a), double **(&b), int r1, int c1, int r2, int c2) {
            double **c = NULL;
            init_2D_mat(c, r1, c2);

            IFOR(i, 0, r1)
                IFOR(j, 0, c2)
                    IFOR(k, 0, c1)
                        c[i][j] = c[i][j] + a[i][k] * b[k][j];

            return c;
        }


        double *vector_add(double *(&a), double *(&b), int row) {
            double *add = NULL;
            init_1D_mat(add, row);

            IFOR(i, 0, row)
                add[i] = a[i] + b[i];

            return add;
        }

        double **add_2D_mat_1D_mat(double **a, double *b, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                    a[i][j] += b[j];
            return a;
        }

        double **diff_2D_mat_1D_mat(double **a, double *b, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                    a[i][j] -= b[i];
            return a;
        }


        double **scalar_add_2D_mat(double **mat, int scalar, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                    mat[i][j] += scalar;
            return mat;
        }

        double **scalar_divide_2D_mat(double **mat, double scalar, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                    mat[i][j] /= scalar;
            return mat;
        }

        double **mat_transpose(double **a, int r, int c) {
            double **trans;
            init_2D_mat(trans, c, r);
            IFOR(i, 0, r)
                IFOR(j, 0, c)
                    trans[i][j] = a[j][i];
            return trans;
        }

        double **scalar_multiply_2D_mat(double **mat, int scalar, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                    mat[i][j] *= scalar;
            return mat;
        }


        double *scalar_divide_1D_mat(double *mat, int scalar, int r) {
            UFOR(i, 0, r)
                mat[i] /= scalar;
            return mat;
        }

        double *scalar_multiply_1D_mat(double *mat, int scalar, int r) {
            UFOR(i, 0, r)
            {
                mat[i] *= scalar;
            }
            return mat;
        }



        double *sum_across_2nd_dim(double **a, int r, int c) {
            double *sum;
            init_1D_mat(sum, r);
            UFOR(i, 0, r)
            {
                sum[i] = 0;
                UFOR(j, 0, c)
                    sum[i] += a[i][j];
            }
            return sum;
        }

        double **element_wise_multiply(double **a, double **b, int r, int c) {
            UFOR(i, 0, r)
                UFOR(j, 0, c)
                a[i][j] *= b[i][j];
            return a;
        }

};
