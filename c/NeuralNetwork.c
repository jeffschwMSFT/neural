// neural_network.c
// A C translation of the provided C# NeuralNetwork

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "NeuralNetwork.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NN_ASSERT
#define NN_ASSERT(cond, msg) do { if(!(cond)) { fprintf(stderr, "NN ASSERT: %s\n", msg); exit(1);} } while(0)
#endif

// Jagged arrays in C:
// Weight[layer][neuron][incoming]
// Bias[layer][neuron] (single float)
typedef struct NeuralNetwork {
    // topology
    int input_n;
    int output_n;
    int layers;              // (hidden_count + 1) edges: input->hidden[0], ..., lastHidden->output
    int *neurons;            // size = layers; neurons in destination of each weight "edge"
    int *incoming;           // size = layers; incoming connections count for that edge

    // params
    float ***Weight;         // [layers][neuron][incoming]
    float  **Bias;           // [layers][neuron]  (scalar each)

    // updates (accumulated)
    float ***dW;             // same shape as Weight
    float  **dB;             // same shape as Bias

    float learning_rate;
} NeuralNetwork;

// ---------- Implementation ----------

// -- RNG helpers (portable) --
static inline float frand01(void) {
    return (float)rand() / (float)RAND_MAX; // [0,1]
}

static float rand_gaussian(void) {
    // Box-Muller; returns ~N(0,1)
    float u1 = frand01();
    while (u1 == 0.0f) u1 = frand01();
    float u2 = frand01();
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    return r * cosf(theta);
}

void nn_seed(unsigned int seed) {
    srand(seed);
}

// -- Allocation helpers --
static float*** alloc_weights(int layers, const int *neurons, const int *incoming) {
    float ***W = (float***)calloc(layers, sizeof(float**));
    NN_ASSERT(W, "alloc W");
    for (int L = 0; L < layers; ++L) {
        W[L] = (float**)calloc(neurons[L], sizeof(float*));
        NN_ASSERT(W[L], "alloc W[L]");
        for (int n = 0; n < neurons[L]; ++n) {
            W[L][n] = (float*)calloc(incoming[L], sizeof(float));
            NN_ASSERT(W[L][n], "alloc W[L][n]");
        }
    }
    return W;
}
static float** alloc_bias(int layers, const int *neurons) {
    float **B = (float**)calloc(layers, sizeof(float*));
    NN_ASSERT(B, "alloc B");
    for (int L = 0; L < layers; ++L) {
        B[L] = (float*)calloc(neurons[L], sizeof(float));
        NN_ASSERT(B[L], "alloc B[L]");
    }
    return B;
}
static void free_weights(int layers, const int *neurons, float ***W) {
    if (!W) return;
    for (int L = 0; L < layers; ++L) {
        if (W[L]) {
            for (int n = 0; n < neurons[L]; ++n) free(W[L][n]);
            free(W[L]);
        }
    }
    free(W);
}
static void free_bias(int layers, float **B) {
    if (!B) return;
    for (int L = 0; L < layers; ++L) free(B[L]);
    free(B);
}

// -- Math helpers (mirror C#) --
static float* relu(const float *a, int len) {
    float *r = (float*)malloc(len * sizeof(float));
    NN_ASSERT(r, "relu alloc");
    for (int i = 0; i < len; ++i) r[i] = a[i] > 0.0f ? a[i] : 0.0f;
    return r;
}
static float* d_relu(const float *a, int len) {
    float *r = (float*)malloc(len * sizeof(float));
    NN_ASSERT(r, "d_relu alloc");
    for (int i = 0; i < len; ++i) r[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;
    return r;
}
static float* softmax(const float *a, int len) {
    float *r = (float*)malloc(len * sizeof(float));
    NN_ASSERT(r, "softmax alloc");
    float maxv = -FLT_MAX;
    for (int i = 0; i < len; ++i) if (a[i] > maxv) maxv = a[i];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        r[i] = expf(a[i] - maxv);
        sum += r[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int i = 0; i < len; ++i) r[i] /= sum;
    return r;
}

static float dot_cap(const float *a, const float *b, int len) {
    float result = 0.0f;
    int sign = 1;
    for (int i = 0; i < len; ++i) {
        result += a[i] * b[i];
        if (result == result && result != INFINITY && result != -INFINITY) {
            sign = (result < 0.0f) ? -1 : 1;
        } else break;
    }
    if (isinf(result) || (isnan(result) && sign > 0)) result = FLT_MAX;
    else if (isinf(-result) || (isnan(result) && sign < 0)) result = -FLT_MAX;
    return result;
}

// result = A^T dot b
// A: [rows=b_len][cols=m]  (nn.Weight[next_layer] shape)
// returns float[m]
static float* dot_first_param_T(float **A, int rows, int m, const float *b) {
    float *r = (float*)calloc(m, sizeof(float));
    NN_ASSERT(r, "dot_first_param_T alloc");
    for (int i = 0; i < m; ++i) {
        float v = 0.0f;
        int sign = 1;
        for (int j = 0; j < rows; ++j) {
            v += A[j][i] * b[j];
            if (v == v && v != INFINITY && v != -INFINITY) sign = (v < 0.0f) ? -1 : 1;
            else break;
        }
        if (isinf(v) || (isnan(v) && sign > 0)) v = FLT_MAX;
        else if (isinf(-v) || (isnan(v) && sign < 0)) v = -FLT_MAX;
        r[i] = v;
    }
    return r;
}

// result += a * b^T
// result shape: [len_a][len_b]
static void dot_second_param_T_accum(const float *a, int len_a, const float *b, int len_b, float **result) {
    for (int i = 0; i < len_a; ++i) {
        for (int j = 0; j < len_b; ++j) {
            result[i][j] += a[i] * b[j];
        }
    }
}

static void vec_sub_inplace(float *a, const float *b, int len) {
    for (int i = 0; i < len; ++i) a[i] = a[i] - b[i];
}
static float* vec_sub_new(const float *a, const float *b, int len) {
    float *r = (float*)malloc(len * sizeof(float));
    NN_ASSERT(r, "vec_sub_new");
    for (int i = 0; i < len; ++i) r[i] = a[i] - b[i];
    return r;
}
static void vec_mul_scalar_inplace(float *a, int len, float v) {
    for (int i = 0; i < len; ++i) a[i] *= v;
}
static void vec_mul_elem_inplace(float *a, const float *b, int len) {
    for (int i = 0; i < len; ++i) a[i] *= b[i];
}

// -- Initialization helpers --
static float init_bias(NeuralBiasInitialization binit) {
    switch (binit) {
        case Bias_Random_Uniform_NegHalf_PosHalf:      return frand01() - 0.5f;
        case Bias_Random_Uniform_NegOneTenth_PosOneTenth: return (frand01() * 0.2f) - 0.1f;
        case Bias_Zero:                                 return 0.0f;
        case Bias_SmallConstant_OneTenth:               return 0.1f;
        case Bias_SmallConstant_OneHundredth:           return 0.01f;
        default: NN_ASSERT(0, "unknown bias init"); return 0.0f;
    }
}
static float init_weight(NeuralWeightInitialization winit, int n_in, int n_out) {
    float limit = sqrtf(6.0f / (float)(n_in + n_out));
    float std2  = sqrtf(2.0f / (float)n_in);
    float std1  = sqrtf(1.0f / (float)n_in);
    switch (winit) {
        case Random_Uniform_NegHalf_PosHalf: return frand01() - 0.5f;
        case Random_Uniform_NegOne_PosOne:   return (frand01() * 2.0f) - 1.0f;
        case Xavier:                         return ((frand01() * 2.0f) - 1.0f) * limit;
        case He:                              return rand_gaussian() * std2;
        case LeCun:                           return rand_gaussian() * std1;
        default: NN_ASSERT(0, "unknown weight init"); return 0.0f;
    }
}

// -- Create/Free --
NeuralNetwork* nn_create(const NeuralOptions *opt) {
    NN_ASSERT(opt, "options null");
    NN_ASSERT(opt->InputNumber > 0, "InputNumber > 0");
    NN_ASSERT(opt->OutputNumber > 0, "OutputNumber > 0");
    NN_ASSERT(opt->LearningRate > 0.0f, "LearningRate > 0");

    NeuralNetwork *nn = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    NN_ASSERT(nn, "alloc nn");

    nn->learning_rate = opt->LearningRate;
    nn->input_n = opt->InputNumber;
    nn->output_n = opt->OutputNumber;

    int hidden_count = opt->HiddenLayerCount;
    const int *hidden = opt->HiddenLayerNumber;

    // Edges count = hidden_count + 1
    nn->layers = hidden_count + 1;
    nn->neurons  = (int*)calloc(nn->layers, sizeof(int));
    nn->incoming = (int*)calloc(nn->layers, sizeof(int));
    NN_ASSERT(nn->neurons && nn->incoming, "alloc topo");

    for (int L = 0; L < nn->layers; ++L) {
        int dst_neurons = (L < hidden_count) ? hidden[L] : nn->output_n;
        int src_neurons = (L == 0) ? nn->input_n : hidden[L - 1];
        NN_ASSERT(dst_neurons > 0 && src_neurons > 0, "layer sizes > 0");
        nn->neurons[L]  = dst_neurons;
        nn->incoming[L] = src_neurons;
    }

    nn->Weight = alloc_weights(nn->layers, nn->neurons, nn->incoming);
    nn->Bias   = alloc_bias(nn->layers, nn->neurons);
    nn->dW     = alloc_weights(nn->layers, nn->neurons, nn->incoming);
    nn->dB     = alloc_bias(nn->layers, nn->neurons);

    // initialize params
    for (int L = 0; L < nn->layers; ++L) {
        int n_out = nn->neurons[L];
        int n_in  = nn->incoming[L];
        for (int n = 0; n < n_out; ++n) {
            nn->Bias[L][n] = init_bias(opt->BiasInitialization);
            for (int i = 0; i < n_in; ++i) {
                nn->Weight[L][n][i] = init_weight(opt->WeightInitialization, n_in, n_out);
            }
        }
    }

    return nn;
}

void nn_destroy(NeuralNetwork *nn) {
    if (!nn) return;
    free_weights(nn->layers, nn->neurons, nn->Weight);
    free_bias(nn->layers, nn->Bias);
    free_weights(nn->layers, nn->neurons, nn->dW);
    free_bias(nn->layers, nn->dB);
    free(nn->neurons);
    free(nn->incoming);
    free(nn);
}

// -- Evaluate (forward) --
NeuralOutput nn_evaluate(const NeuralNetwork *nn, const float *input) {
    NN_ASSERT(nn && input, "eval args");
    NN_ASSERT(nn->input_n > 0, "input_n");

    NeuralOutput out = {0};
    out.Result = -1;
    out.layers = nn->layers;
    out.neurons_per_layer = (int*)malloc(nn->layers * sizeof(int));
    NN_ASSERT(out.neurons_per_layer, "alloc neurons_per_layer");

    out.Input = (float*)malloc(nn->input_n * sizeof(float));
    NN_ASSERT(out.Input, "alloc out.Input");
    memcpy(out.Input, input, nn->input_n * sizeof(float));

    out.Z = (float**)calloc(nn->layers, sizeof(float*));
    out.A = (float**)calloc(nn->layers, sizeof(float*));
    NN_ASSERT(out.Z && out.A, "alloc Z/A");

    for (int L = 0; L < nn->layers; ++L) {
        int n_out = nn->neurons[L];
        int n_in  = nn->incoming[L];
        out.neurons_per_layer[L] = n_out;

        out.Z[L] = (float*)calloc(n_out, sizeof(float));
        NN_ASSERT(out.Z[L], "alloc Z[L]");

        const float *prev = (L == 0) ? out.Input : out.A[L - 1];

        // Z[L][neuron] = W[L][neuron] dot prev + B[L][neuron]
        for (int n = 0; n < n_out; ++n) {
            out.Z[L][n] = dot_cap(nn->Weight[L][n], prev, n_in) + nn->Bias[L][n];
        }

        // A[L] = ReLU(Z[L]) for hidden, Softmax(Z[L]) for last
        if (L < nn->layers - 1) out.A[L] = relu(out.Z[L], n_out);
        else                    out.A[L] = softmax(out.Z[L], n_out);
    }

    // determine result + copy probabilities
    int last = nn->layers - 1;
    int m = nn->neurons[last];
    out.Probabilities = (float*)malloc(m * sizeof(float));
    NN_ASSERT(out.Probabilities, "alloc probabilities");
    float maxv = -FLT_MAX;
    int argmax = -1;
    for (int i = 0; i < m; ++i) {
        out.Probabilities[i] = out.A[last][i];
        if (out.A[last][i] > maxv) { maxv = out.A[last][i]; argmax = i; }
    }
    out.Result = argmax;

    return out;
}

void nn_destroy_output(NeuralOutput *out) {
    if (!out) return;
    free(out->Probabilities);
    free(out->Input);
    if (out->Z) {
        for (int L = 0; L < out->layers; ++L) free(out->Z[L]);
        free(out->Z);
    }
    if (out->A) {
        for (int L = 0; L < out->layers; ++L) free(out->A[L]);
        free(out->A);
    }
    free(out->neurons_per_layer);
    memset(out, 0, sizeof(*out));
}

// -- Learn (backprop) --
void nn_learn_index(NeuralNetwork *nn, const NeuralOutput *out, int preferredResult) {
    NN_ASSERT(nn && out, "learn_index args");
    NN_ASSERT(preferredResult >= 0 && preferredResult < nn->output_n, "preferredResult range");
    int last = nn->layers - 1;
    int m = nn->neurons[last];

    float *Y = (float*)calloc(m, sizeof(float));
    NN_ASSERT(Y, "alloc Y");
    Y[preferredResult] = 1.0f;
    nn_learn_vector(nn, out, Y);
    free(Y);
}

void nn_learn_vector(NeuralNetwork *nn, const NeuralOutput *out, const float *preferredResults) {
    NN_ASSERT(nn && out && preferredResults, "learn_vector args");
    NN_ASSERT(out->Result >= 0 && out->Input && out->A && out->Z, "output validity");
    NN_ASSERT(nn->output_n == nn->neurons[nn->layers - 1], "topology check");

    int last = nn->layers - 1;

    // dZ_last = 2 * (A_last - Y)
    float *dZ_last = vec_sub_new(out->A[last], preferredResults, nn->neurons[last]);
    vec_mul_scalar_inplace(dZ_last, nn->neurons[last], 2.0f);

    // dW_last += dZ_last * A_prev^T
    const float *A_prev = (nn->layers == 1) ? out->Input : out->A[last - 1];
    dot_second_param_T_accum(dZ_last, nn->neurons[last], A_prev, nn->incoming[last], nn->dW[last]);

    // dB_last += dZ_last
    for (int i = 0; i < nn->neurons[last]; ++i) nn->dB[last][i] += dZ_last[i];

    // backprop hidden layers
    float *dZ_next = dZ_last;
    for (int L = last - 1; L >= 0; --L) {
        // dZ_current = (W[L+1]^T dot dZ_next) .* dReLU(Z[L])
        float *tmp = dot_first_param_T(nn->Weight[L + 1],
                                       nn->neurons[L + 1],
                                       nn->incoming[L + 1],
                                       dZ_next);
        float *gprime = d_relu(out->Z[L], nn->neurons[L]);
        vec_mul_elem_inplace(tmp, gprime, nn->neurons[L]);
        free(gprime);

        // dW[L] += dZ_current * A_prev^T
        const float *A_prevL = (L == 0) ? out->Input : out->A[L - 1];
        dot_second_param_T_accum(tmp, nn->neurons[L], A_prevL, nn->incoming[L], nn->dW[L]);

        // dB[L] += dZ_current
        for (int i = 0; i < nn->neurons[L]; ++i) nn->dB[L][i] += tmp[i];

        // propagate
        free(dZ_next);
        dZ_next = tmp;
    }
    free(dZ_next);

    // Apply updates and clear accumulators
    for (int L = 0; L < nn->layers; ++L) {
        int n_out = nn->neurons[L];
        int n_in  = nn->incoming[L];
        for (int n = 0; n < n_out; ++n) {
            // W = W - alpha * dW
            for (int i = 0; i < n_in; ++i) {
                nn->Weight[L][n][i] -= nn->learning_rate * nn->dW[L][n][i];
                nn->dW[L][n][i] = 0.0f;
            }
            // B = B - alpha * dB
            nn->Bias[L][n] -= nn->learning_rate * nn->dB[L][n];
            nn->dB[L][n] = 0.0f;
        }
    }
}
