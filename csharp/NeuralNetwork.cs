using System;
using System.IO;
using System.Security.Cryptography;

// A lot of reference material was used to generate this code.  Here are a few of the most useful materials I found.
//   https://en.wikipedia.org/wiki/Mathematics_of_artificial_neural_networks
//   https://www.youtube.com/watch?v=aircAruvnKk (series of videos)
//   http://neuralnetworksanddeeplearning.com/index.html
//     https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
//   https://www.youtube.com/watch?v=w8yWXqWQYmU
//     https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook

namespace Learning
{
    public struct NeuralOutput
    {
        public int Result;
        public float[] Probabilities;

        #region internal
        internal float[] Input;
        internal float[][] Z;
        internal float[][] A;
        #endregion
    }

    public enum NeuralWeightInitialization
    {
        Random_Uniform_NegHalf_PosHalf = 0,  // default
        Random_Uniform_NegOne_PosOne = 1,
        Xavier = 2, //(sqrt(6 / (n_in + n_out)))
        He = 3, // (sqrt(2 / n_in))
        LeCun = 4 // (1 / sqrt(n_in))
    }

    public enum NeuralBiasInitialization
    {
        Random_Uniform_NegHalf_PosHalf = 0,  // default
        Random_Uniform_NegOneTenth_PosOneTenth = 1,
        Zero = 2,
        SmallConstant_OneTenth = 3,
        SmallConstant_OneHundredth = 4
    }

    public struct NeuralOptions
    {
        public int InputNumber;         // number of options in the input layer (eg. 10)
        public int OutputNumber;        // number of options in the output layer (eg. 10)
        public int[] HiddenLayerNumber; // an array containing how many hidden layers (Length) and how many options per layer
                                        // (eg. int[] { 10 } == one hidden layer of 10 options)
                                        // (default 1 hidden layer with 10)
        public float LearningRate;      // neural network learning rate (alpha) (eg. 0.15f)
        public NeuralWeightInitialization WeightInitialization; // how to initialize the weights and bias' (default Random_Uniform_NegHalf_PosHalf)
        public NeuralBiasInitialization BiasInitialization; // how to initialize the bias' (default Zero)
    }

    public class NeuralNetwork
    {
        public NeuralNetwork(NeuralOptions options) : this()
        {
            // validate options
            if (options.InputNumber <= 0) throw new Exception("must provide a positive number of input");
            if (options.OutputNumber <= 0) throw new Exception("must provide a positive number of output");
            if (options.LearningRate <= 0) throw new Exception("must provide a positive learning rate");
            if (options.HiddenLayerNumber == null) options.HiddenLayerNumber = new int[] { 10 };
            for (int i = 0; i < options.HiddenLayerNumber.Length; i++)
            {
                if (options.HiddenLayerNumber[i] <= 0) throw new Exception($"hidden layer {i} must be a positive number");
            }

            // initialize number of layers (input + hidden.Length + output - 1)
            Weight = new float[options.HiddenLayerNumber.Length + 1][][];
            Bias = new float[options.HiddenLayerNumber.Length + 1][][];

            // create the weights and bias'
            using (var rand = RandomNumberGenerator.Create())
            {
                for (int layer = 0; layer < options.HiddenLayerNumber.Length + 1; layer++)
                {
                    // initialize neurons in layer
                    var numNeuronsInLayer = layer < options.HiddenLayerNumber.Length ? options.HiddenLayerNumber[layer] : options.OutputNumber;
                    Weight[layer] = new float[numNeuronsInLayer][];
                    Bias[layer] = new float[numNeuronsInLayer][];

                    for (int neuron = 0; neuron < numNeuronsInLayer; neuron++)
                    {
                        // initialize connections into layer
                        var numIncomingConnections = (layer == 0) ? options.InputNumber : options.HiddenLayerNumber[layer - 1];
                        Weight[layer][neuron] = new float[numIncomingConnections];

                        // bias'
                        var bias = 0f;
                        switch (options.BiasInitialization)
                        {
                            case NeuralBiasInitialization.Random_Uniform_NegHalf_PosHalf:
                                bias = (GetRandom(rand) * 1f) - 0.5f;
                                break;
                            case NeuralBiasInitialization.Random_Uniform_NegOneTenth_PosOneTenth:
                                bias = (GetRandom(rand) * 0.2f) - 0.1f;
                                break;
                            case NeuralBiasInitialization.Zero:
                                bias = 0f;
                                break;
                            case NeuralBiasInitialization.SmallConstant_OneTenth:
                                bias = 0.1f;
                                break;
                            case NeuralBiasInitialization.SmallConstant_OneHundredth:
                                bias = 0.01f;
                                break;
                            default:
                                throw new Exception("unknown bias initialization");
                        }
                        Bias[layer][neuron] = new float[1] { bias };

                        // initialize weights with random values
                        var limit = (float)Math.Sqrt(6f / (float)(numIncomingConnections + numNeuronsInLayer));
                        var stdDev2 = (float)Math.Sqrt(2f / (float)numIncomingConnections);
                        var stdDev1 = (float)Math.Sqrt(1f / (float)numIncomingConnections);
                        for (int i = 0; i < numIncomingConnections; i++)
                        {
                            var weight = 0f;
                            switch (options.WeightInitialization)
                            {
                                case NeuralWeightInitialization.Random_Uniform_NegHalf_PosHalf:
                                    weight = (GetRandom(rand) * 1f) - 0.5f;
                                    break;
                                case NeuralWeightInitialization.Random_Uniform_NegOne_PosOne:
                                    weight = (GetRandom(rand) * 2f) - 1f;
                                    break;
                                case NeuralWeightInitialization.Xavier:
                                    weight = ((GetRandom(rand) * 2f) - 1f) * limit;
                                    break;
                                case NeuralWeightInitialization.He:
                                    weight = GetGaussianRandom(rand) * stdDev2;
                                    break;
                                case NeuralWeightInitialization.LeCun:
                                    weight = GetGaussianRandom(rand) * stdDev1;
                                    break;
                                default:
                                    throw new Exception("unknown weight initialization");
                            }
                            Weight[layer][neuron][i] = weight;
                        }
                    }
                }
            } // using (rand)

            Initialize(options);
        }

        public float LearningRate { get; private set; }
        public int InputNumber { get; private set; }
        public int OutputNumber { get; private set; }

        public NeuralOutput Evaluate(float[] input)
        {
            // forward propagate

            // validate
            if (input == null || input.Length != InputNumber) throw new Exception("invalid input");

            // output
            var output = new NeuralOutput()
            {
                // result
                Result = -1,

                // internal details needed for backward propagate
                Input = new float[input.Length],
                Z = new float[Weight.Length][],
                A = new float[Weight.Length][]
            };

            // save input
            Array.Copy(input, output.Input, input.Length);

            // High level: 
            //  Each neuron in layer 1 is connected to each neuron in layer 2, 
            //  and each neuron in layer 2 is connected to each neuron in layer 3,
            //  and so on.
            //  In between each layer there are Weights and Bias' that are applied in this formula.
            //    = ReLU(w0*a0 + w1*a1 + ... + wn*an + Bias)
            //  The final result is calculated via a soft max and the index with the highest value
            //  is the result.

            /*
             Formulas
              Z[1]=W[1]X+b[1]
              A[1]=gReLU(Z[1]))
              Z[2]=W[2]A[1]+b[2]
              A[2]=gsoftmax(Z[2])
             */

            // progress through layers
            for (int layer = 0; layer < Weight.Length; layer++)
            {
                // first round
                //    Z0 = [Weight0] dot [input] + [Bias0]
                // second round 
                //    Z1 = [Weight1] dot [A0] + [Bias1]
                // last round
                //    Zn = [Weightn] dot [An-1] + [Biasn]
                output.Z[layer] = new float[Weight[layer].Length];
                for (var neuron = 0; neuron < Weight[layer].Length; neuron++)
                { 
                    output.Z[layer][neuron] = Dot(Weight[layer][neuron], (layer == 0) ? input : output.A[layer - 1]) + Bias[layer][neuron][0]; 
                }

                // first round
                //    A0 = ReLU(Z0)
                // second round
                //    A1 = ReLU(Z1)
                // last round
                //    An = Softmax(Zn)
                if (layer < Weight.Length - 1) output.A[layer] = ReLU(output.Z[layer]);
                else output.A[layer] = Softmax(output.Z[layer]);
            }

            // output is the index with the greatest value in An
            var max = Single.MinValue;
            output.Probabilities = new float[output.A[output.A.Length - 1].Length];
            for(int i=0; i < output.A[output.A.Length-1].Length; i++)
            {
                // copy all probabilities
                output.Probabilities[i] = output.A[output.A.Length - 1][i];
                // find the largest
                if (max < output.A[output.A.Length - 1][i])
                {
                    max = output.A[output.A.Length - 1][i];
                    output.Result = i;
                }
            }

            return output;
        }

        public void Learn(NeuralOutput output, int preferredResult)
        {
            if (preferredResult < 0 || preferredResult >= OutputNumber) throw new Exception("invalid preferred result");
            // create the output array with the right element set
            var preferredResults = new float[OutputNumber];
            preferredResults[preferredResult] = 1f;
            // learn
            Learn(output, preferredResults);
        }

        public void Learn(NeuralOutput output, float[] preferredResults)
        { 
            // backward propagate

            // validate
            if (output.Result < 0 || output.Result >= OutputNumber ||
                output.Input == null || output.A == null || output.Z == null ||
                output.Input.Length != InputNumber) throw new Exception("invalid output");
            if (preferredResults == null || preferredResults.Length != OutputNumber) throw new Exception("invalid preferred results");

            // high level:
            //  compute the partial derivatives of
            //    dC/dweight and dC/dbias
            //    where C = (y-An)^2 (where Y is the best outcome)
            //  looking for local minimum of the complex function - Gradient descent
            // error = dC/dAn*dReLU(Zn)
            //  measures how much C changes at An (dC/dAn) and how much the activation function changes at Zn
            // update the weights and bias' to learn for the next round (per mini batch).
            // The algorithm works backwards from the last layer to the first (as the name suggests)

            /*
             Formulas
              dZ[2]=A[2]−Y
              dW[2]=1mdZ[2]A[1]T
              dB[2]=1mΣdZ[2]
              dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])
              dW[1]=1mdZ[1]A[0]T
              dB[1]=1mΣdZ[1]
             */

            // init
            var layer = Weight.Length - 1;
            float[] dZlast;

            // create the output array with the right element set
            var Y = preferredResults;

            // float[] dZ = 2 * (A - Y)
            dZlast = Subtract(output.A[layer], Y);
            Multiply(2f, ref dZlast);

            // float[][] dW = dZ * A(-1)T
            DotSecondParamT(dZlast, output.A[layer - 1], ref AggregatedUpdate.dW[layer]);

            // float dB = dZ (update the aggregated updates)
            for (int i=0; i<dZlast.Length; i++) AggregatedUpdate.dB[layer][i] += dZlast[i];

            // compute the rest in context of these values, backwards
            for (layer = Weight.Length - 2; layer >= 0; layer--)
            {
                // float[] dZcurrent = W(+1).T dot dZlast * g`(Z)
                var dZcurrent = DotFirstParamT(Weight[layer + 1], dZlast);
                Multiply(ref dZcurrent, dOfReLU(output.Z[layer]));

                // float[][] dW = dZ dot A(-1)T
                DotSecondParamT(dZcurrent, (layer == 0) ? output.Input : output.A[layer - 1], ref AggregatedUpdate.dW[layer]);

                // float dB = dZ (update the aggregated updates)
                for (int i = 0; i < dZcurrent.Length; i++) AggregatedUpdate.dB[layer][i] += dZcurrent[i];

                // pass dZcurrent calculation to next layer
                dZlast = dZcurrent;
            }

            // update the weights and bias'
            for (layer = 0; layer < Weight.Length; layer++)
            {
                for (int neuron = 0; neuron < Weight[layer].Length; neuron++)
                {
                    // W = W - alpha * dW
                    Multiply(LearningRate, ref AggregatedUpdate.dW[layer][neuron]);
                    Subtract(ref Weight[layer][neuron], AggregatedUpdate.dW[layer][neuron]);
                    // clear
                    for (int i = 0; i < AggregatedUpdate.dW[layer][neuron].Length; i++) AggregatedUpdate.dW[layer][neuron][i] = 0f;

                    // B = B - alpha * dB
                    Bias[layer][neuron][0] = Bias[layer][neuron][0] - (LearningRate * AggregatedUpdate.dB[layer][neuron]);
                    // clear
                    AggregatedUpdate.dB[layer][neuron] = 0f;
                }
           }
        }

        #region private
        // example
        //  10 input, 16 hidden layer, 5 outputs
        //
        //        | between               | between              |
        //        | layer 0 & 1           | layer 1 & 2          |
        // -------------------------------------------------------
        // weight | {16, 10}              | {5, 16}              |
        //        | 16 neurons in layer 1 | 5 neurons in output  |
        //        | each connected to 10  | each connected to 16 |
        //        | neurons in layer 0    | neurons in layer 1   |
        // -------------------------------------------------------
        // bias   | {16, 1}               | {5, 1}               |
        //        | 16 neurons in layer 1 | 5 neurons in output  |
        //        | each connected to 10  | each connected to 16 |
        //        | neurons in layer 0    | neurons in layer 1   |
        // -------------------------------------------------------
        //
        // layer 0    layer 1     layer 2
        //  0 --------- 0    ...    0
        //    \________ 1           1
        //  1  \_______ 2           ...
        //  ...         ...         5
        //  10 -----    ...
        //              \_ 15   

        // Dimensions
        //  0 - layers (eg. 0 [input to hidden], 1 - [hidden to output])
        //  1 - number of neurons in layer (eg. 0 - 10 [hidden], 1 - 10 [output]) (skip input, start at first hidden to end)
        //  2 - number of connections into layer (eg. 0 - 784 [input to hidden], 1 - 10 [hidden to output])  (start at input to end, skip output)
        private float[][][] Weight;
        private float[][][] Bias;

        // batch up updates to apply (the average smooths out learning)
        struct NeuralUpdate
        {
            public float[][][] dW;
            public float[][] dB;
        }
        private NeuralUpdate AggregatedUpdate;

        private void Initialize(NeuralOptions options)
        {
            // initialize
            LearningRate = options.LearningRate;
            InputNumber = options.InputNumber;
            OutputNumber = options.OutputNumber;

            // used for updates
            AggregatedUpdate = new NeuralUpdate()
            {
                dW = new float[Weight.Length][][],
                dB = new float[Bias.Length][]
            };
            for (var i = 0; i < AggregatedUpdate.dW.Length; i++)
            {
                AggregatedUpdate.dW[i] = new float[Weight[i].Length][];
                AggregatedUpdate.dB[i] = new float[Bias[i].Length];
                for (var j = 0; j < AggregatedUpdate.dW[i].Length; j++)
                {
                    AggregatedUpdate.dW[i][j] = new float[Weight[i][j].Length];
                }
            }
        }

        private static float GetRandom(RandomNumberGenerator rand)
        {
            Span<byte> int32buffer = stackalloc byte[4];
            rand.GetNonZeroBytes(int32buffer);
            // ensure positive
            int32buffer[3] &= 0x7f;
            var number = BitConverter.ToInt32(int32buffer);
            // get a random float between 0f and 1f
            return ((float)number / (float)Int32.MaxValue);
        }

        private float Sigmoid(float x)
        {
            // will normalize the value between 0 and 1
            return 1f / (1f + (float)Math.Exp(-1f * x));
        }

        private float GetGaussianRandom(RandomNumberGenerator rand)
        {
            // return a random number that fits a gaussian distribution between -phi and phi
            // using the Box-Muller transform
            var u1 = GetRandom(rand);
            while (u1 == 0f) u1 = GetRandom(rand);
            var u2 = GetRandom(rand);
            var z0 = (float)(Math.Sqrt(-2f * Math.Log(u1)) * Math.Cos(2f * Math.PI * u2));
            // the second value is discarded
            // var z1 = (float)(Math.Sqrt(-2f * Math.Log(u1)) * Math.Sin(2f * Math.PI * u2));
            return z0;
        }

        private static NeuralNetwork SelectNetworkByDistribution(NeuralNetwork[] networks, float[] distribution, RandomNumberGenerator rand)
        {
            var r = GetRandom(rand);
            for (int i = 0; i < distribution.Length; i++)
            {
                if (r < distribution[i]) return networks[i];
            }
            return networks[^1]; // fallback
        }

        private static float GetMutation(float probability, float strength, RandomNumberGenerator rand)
        {
            if (probability <= 0f || strength <= 0f) return 0f;

            // apply mutation with a small chance
            if (GetRandom(rand) < probability)
            {
                // (-1f to 1f) * mutation strength
                return ((GetRandom(rand) * 2f) - 1f) * strength;
            }

            return 0f;
        }
        #endregion

        #region protected
        // protected to enable testing

        // constructor for Load
        protected NeuralNetwork()
        {
            // init
            //Updates = new List<NeuralUpdate>();
        }

        protected static float[] ReLU(float[] a)
        {
            // if x <= 0 : 0
            //    x >  0 : x
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) result[i] = Math.Max(0f, a[i]);
            return result;
        }

        protected static float[] dOfReLU(float[] a)
        {
            // derivative of ReLU
            // if x <= 0 : 0
            //    x >  0 : 1
            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++) result[i] = (a[i] > 0 ? 1 : 0);
            return result;
        }

        protected static float[] Softmax(float[] a)
        {
            // = foreach(var x in a) r += e^(x-max) / sum(e^x-max)
            var sum = 0f;
            var max = Single.MinValue;
            var result = new float[a.Length];

            // max
            for (int i = 0; i < a.Length; i++) if (max < a[i]) max = a[i];
            // sum(e^(x-max))
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = (float)Math.Pow(Math.E, a[i] - max);
                sum += result[i];
            }
            // e^(x-max) / sum(e^x-max)
            for (int i = 0; i < a.Length; i++) result[i] = result[i] / sum;

            return result;
        }

        protected static float Dot(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("lengths must match");
            var result = 0f;
            var sign = 1;
            for(int i=0; i<a.Length; i++)
            {
                result += (a[i] * b[i]);

                // check where the result was heading (unwrap the function calls IsNan, IsInfinity)
                if (result == result &&
                    result != Single.PositiveInfinity &&
                    result != Single.NegativeInfinity) sign = result < 0 ? -1 : 1;
                // done with the loop
                else break;
            }

            // cap
            if (Single.IsPositiveInfinity(result) || (Single.IsNaN(result) && sign > 0)) result = Single.MaxValue;
            else if (Single.IsNegativeInfinity(result) || (Single.IsNaN(result) && sign < 0)) result = Single.MinValue;

            return result;
        }

        protected static float[] DotFirstParamT(float[][] a, float[] b)
        {
            if (a == null || a.Length == 0 ||
                a.Length != b.Length) throw new Exception("invalid array lengths");

            // fake [[,,,],[,,,]] dot [[],[],[]]
            // first parameter is transposed
            var result = new float[a[0].Length];
            for (int i = 0; i < a[0].Length; i++)
            {
                var sign = 1;
                for (int j = 0; j < b.Length; j++)
                {
                    // keeping the column stable (eg. i) while
                    // walking the row (eg. j)
                    result[i] += a[j][i] * b[j];

                    // check where the result was heading (unwrap the function calls IsNan, IsInfinity)
                    if (result[i] == result[i] &&
                        result[i] != Single.PositiveInfinity &&
                        result[i] != Single.NegativeInfinity) sign = result[i] < 0 ? -1 : 1;
                    // done with the loop
                    else break;
                }

                // cap
                if (Single.IsPositiveInfinity(result[i]) || (Single.IsNaN(result[i]) && sign > 0)) result[i] = Single.MaxValue;
                else if (Single.IsNegativeInfinity(result[i]) || (Single.IsNaN(result[i]) && sign < 0)) result[i] = Single.MinValue;
            }
            return result;
        }

        protected static void DotSecondParamT(float[] a, float[] b, ref float[][] result)
        {
            // fake [[],[],[]] dot [[,,,]]
            // second parameter is transposed
            for (int i = 0; i < result.Length; i++)
            {
                for (int j = 0; j < result[i].Length; j++)
                {
                    result[i][j] += a[i] * b[j];
                }
            }
        }

        protected static void Subtract(ref float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("lengths must match");
            for (int i = 0; i < a.Length && i < b.Length; i++)
            {
                a[i] = (a[i] - b[i]);
            }
        }

        protected static float[] Subtract(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("lengths must match");
            var result = new float[a.Length];
            for (int i = 0; i < a.Length && i < b.Length; i++)
            {
                result[i] = (a[i] - b[i]);
            }
            return result;
        }

        protected static void Multiply(float value, ref float[] a)
        {
            for(int i=0; i<a.Length; i++)
            {
                a[i] = a[i] * value;
            }
        }

        protected static void Multiply(ref float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("lengths must match");
            for (int i = 0; i < a.Length && i < b.Length; i++)
            {
                a[i] = a[i] * b[i];
            }
        }
        #endregion
    }
}
