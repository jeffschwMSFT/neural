using System;

namespace NeuralNetworkApp
{
    class Program
    {
        static int Main(string[] args)
        {
            var options = new Learning.NeuralOptions
            {
                InputNumber = 4,
                OutputNumber = 3,
                HiddenLayerNumber = new int[] { 10 },
                LearningRate = 0.15f,
                WeightInitialization = Learning.NeuralWeightInitialization.Xavier,
                BiasInitialization = Learning.NeuralBiasInitialization.Zero
            };

            var nn = new Learning.NeuralNetwork(options);

            float[] input = new float[] { 0.1f, 0.2f, -0.1f, 0.5f };
            var output = nn.Evaluate(input);

            Console.WriteLine($"Result: {output.Result}");
            for (int i = 0; i < options.OutputNumber; ++i)
            {
                Console.WriteLine($"p[{i}]={output.Probabilities[i]}");
            }

            // learn toward class 2
            nn.Learn(output, 2);

            // re-evaluate
            output = nn.Evaluate(input);
            Console.WriteLine($"After learn, Result: {output.Result}");
            for (int i = 0; i < options.OutputNumber; ++i)
            {
                Console.WriteLine($"p[{i}]={output.Probabilities[i]}");
            }

            return 0;
        }
    }
}
