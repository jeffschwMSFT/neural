#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>

// Forward-declare the neural network type (opaque to users)
typedef struct NeuralNetwork NeuralNetwork;

typedef struct {
    int Result;
    float *Probabilities;

    // internal (for Learn)
    float *Input;
    float **Z; // per layer: float[ neurons ]
    float **A; // per layer: float[ neurons ]
    int layers;
    int *neurons_per_layer; // size = layers
} NeuralOutput;

typedef enum {
    Random_Uniform_NegHalf_PosHalf = 0,
    Random_Uniform_NegOne_PosOne   = 1,
    Xavier                         = 2, // sqrt(6/(n_in+n_out)) uniform
    He                             = 3, // N(0, sqrt(2/n_in))
    LeCun                          = 4  // N(0, sqrt(1/n_in))
} NeuralWeightInitialization;

typedef enum {
    Bias_Random_Uniform_NegHalf_PosHalf      = 0,
    Bias_Random_Uniform_NegOneTenth_PosOneTenth = 1,
    Bias_Zero                                 = 2,
    Bias_SmallConstant_OneTenth               = 3,
    Bias_SmallConstant_OneHundredth           = 4
} NeuralBiasInitialization;

typedef struct {
    int InputNumber;
    int OutputNumber;
    int *HiddenLayerNumber; // length hidden_layers
    int HiddenLayerCount;   // number of hidden layers
    float LearningRate;
    NeuralWeightInitialization WeightInitialization;
    NeuralBiasInitialization BiasInitialization;
} NeuralOptions;

// init
void nn_seed(unsigned int seed);

// Create / destroy
NeuralNetwork* nn_create(const NeuralOptions* opt);
void nn_destroy(NeuralNetwork* nn);

// Forward propagation: inputs -> outputs
NeuralOutput nn_evaluate(const NeuralNetwork *nn, const float *input);
void nn_destroy_output(NeuralOutput *out);

// Train (forward + backpropagation + update)
void nn_learn_index(NeuralNetwork *nn, const NeuralOutput *out, int preferredResult);
void nn_learn_vector(NeuralNetwork *nn, const NeuralOutput *out, const float *preferredResults);

#endif // NEURAL_NETWORK_H