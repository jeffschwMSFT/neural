
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"

int main(void) {
    nn_seed(12345);
    int hidden[1] = { 10 };

    NeuralOptions opt = {
        .InputNumber = 4,
        .OutputNumber = 3,
        .HiddenLayerNumber = hidden,
        .HiddenLayerCount = 1,
        .LearningRate = 0.15f,
        .WeightInitialization = Xavier,
        .BiasInitialization = Bias_Zero
    };

    NeuralNetwork *nn = nn_create(&opt);

    float input[4] = {0.1f, 0.2f, -0.1f, 0.5f};
    NeuralOutput out = nn_evaluate(nn, input);

    printf("Result: %d\n", out.Result);
    for (int i = 0; i < opt.OutputNumber; ++i) printf("p[%d]=%f\n", i, out.Probabilities[i]);

    // learn toward class 2
    nn_learn_index(nn, &out, 2);
    nn_destroy_output(&out);

    // re-evaluate
    out = nn_evaluate(nn, input);
    printf("After learn, Result: %d\n", out.Result);
    for (int i = 0; i < opt.OutputNumber; ++i) printf("p[%d]=%f\n", i, out.Probabilities[i]);
    nn_destroy_output(&out);

    nn_destroy(nn);
    return 0;
}
