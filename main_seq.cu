#include <bits/stdc++.h>
#include "neural_network_3layers_seq.cu"


#define IFOR(v, s, e) for(int v = s; v < e; ++v)
#define UFOR(v, s, e) for(unsigned v = s; v < e; v++)

using namespace std;


int main() {
    // Initialize dataset    
    double **train_x, **test_x, *train_y, *test_y;
    Initializer init = Initializer();
    NeuralNetwork NN = NeuralNetwork();

    mu.init_2D_mat(train_x, TRAIN_SAMPLE, DATA_DIM);
    mu.init_2D_mat(test_x, TEST_SAMPLE, DATA_DIM);
    mu.init_1D_mat(train_y, TRAIN_SAMPLE);
    mu.init_1D_mat(test_y, TEST_SAMPLE);

    // load train data
    init.load_data(train_x, train_y, TRAIN_SAMPLE, DATA_DIM + 1, (const char*)"data_500.txt");

    // load test data
    init.load_data(test_x, test_y, TEST_SAMPLE, 1 + DATA_DIM, (const char *)"test_data_100.txt");


    // Initialize FW prop weights
    double **W1 = NULL, **W2 = NULL, *b1 = NULL, *b2 = NULL;
    mu.init_2D_mat(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    mu.init_2D_mat(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);

    mu.init_1D_mat(b1, HIDDEN_LAYER_NODES);
    mu.init_1D_mat(b2, OUTPUT_LAYER_NODES);

    init.init_weights(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    init.init_weights(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);

    init.init_biases(b1, HIDDEN_LAYER_NODES);
    init.init_biases(b2, OUTPUT_LAYER_NODES);

    // Initialize Back Prop delta
    double **dW1, **dW2, *db1, *db2;
    mu.init_2D_mat(dW1, HIDDEN_LAYER_NODES, DATA_DIM);
    mu.init_2D_mat(dW2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    mu.init_1D_mat(db1, HIDDEN_LAYER_NODES);
    mu.init_1D_mat(db2, OUTPUT_LAYER_NODES);

    // Initiaze FW prop return parameters
    double **Z1, **Z2, **A1, **A2;
    mu.init_2D_mat(Z1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    mu.init_2D_mat(A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    mu.init_2D_mat(Z2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    mu.init_2D_mat(A2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);

    // Initiaze Back prop return parameters
    double **dA1, **dA2;
    mu.init_2D_mat(dA1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    mu.init_2D_mat(dA2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);

    // Train neural network
    UFOR(i, 0, EPOCH)   
    {
        cout << "------Iteration: " << i+1 << '\n';
        // Forward Propagation
        NN.forward_prop(train_x, W1, W2, b1, b2, Z1, Z2, A1, A2, TRAIN_SAMPLE);
	
        // Backward Propagation
        NN.back_prop(train_x, train_y, W1, W2, A1, A2, dW1, dW2, dW1, dW2, db1, db2);
	    //cout<<"Finished backprop"<<'\n';
        
        // Parameter Updates
        NN.update_parameter(W1, W2, b1, b2, dW1, dW2, db1, db2);
    }

    // Initiaze FW prop return parameters for test
    double **_Z1 = NULL, **_Z2 = NULL, **_A1 = NULL, **_A2 = NULL;
    mu.init_2D_mat(_Z1, TEST_SAMPLE, HIDDEN_LAYER_NODES);
    mu.init_2D_mat(_A1, TEST_SAMPLE, HIDDEN_LAYER_NODES);
    mu.init_2D_mat(_Z2, TEST_SAMPLE, OUTPUT_LAYER_NODES);
    mu.init_2D_mat(_A2, TEST_SAMPLE, OUTPUT_LAYER_NODES);

    // Test the network
    NN.forward_prop(test_x, W1, W2, b1, b2, _Z1, _Z2, _A1, _A2, TEST_SAMPLE);

    UFOR(i, 0, TEST_SAMPLE)       
    {
        UFOR(j, 0, OUTPUT_LAYER_NODES)       
            if (A2[i][j] < 0.5)
                A2[i][j] = 0;
            else
                A2[i][j] = 1;
    }
    int accurate = 0;
    UFOR(i, 0, TEST_SAMPLE)       
        UFOR(j, 0, OUTPUT_LAYER_NODES)       
            if (A2[i][j] == test_y[i])
                ++accurate;

    double accuracy = ((accurate * 100) / TEST_SAMPLE);
    cout << "\n\nAccuracy of the model is " << accuracy << '\n';

    return 0;
}
