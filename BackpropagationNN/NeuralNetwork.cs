using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    class NeuralNetwork
    {
        private double[][] neurons;
        private double[] biases;
        private double[][][] weights;

        private Random rnd;

        public NeuralNetwork(int[] networkStruc)
        {
            neurons = new double[networkStruc.Length][];
            biases = new double[networkStruc.Length-1];
            weights = new double[networkStruc.Length - 1][][];

            for(int i=0;i<networkStruc.Length;i++)
            {
                neurons[i] = new double[networkStruc[i]];
            }
            InitBiases();
            InitWeights();

        }

        private void InitWeights()
        {
            for(int i=0;i<neurons.Length-1;i++)
            {
                weights[i] = new double[neurons[i].Length][];
                for(int j=0;j<weights[i].Length;j++)
                {
                    weights[i][j] = new double[neurons[i + 1].Length];
                    for(int k=0;k<weights[i][j].Length;k++)
                    {
                        weights[i][j][k] = rnd.NextDouble() - 0.5f;
                    }
                }
            }
        }
        private void InitBiases()
        {
            for (int i = 0; i < neurons.Length - 1; i++)
            {
                biases[i]= rnd.NextDouble() - 0.5f;
            }
        }

        private void Test(double[] input)
        {
            double sumOfLayer=0;
            for(int i=0;i<neurons[0].Length;i++)
            {
                neurons[0][i] = input[i];
            }
            
            for(int i=1;i<neurons.Length;i++)
            {
                for(int j=0;j<neurons[i].Length;j++)
                {
                    for(int k=0;k<neurons[i].Length;k++)
                    {
                        sumOfLayer += neurons[i - 1][k] * weights[i - 1][k][i];

                    }
                    neurons[i][j] = sumOfLayer + biases[i - 1];
                    sumOfLayer = 0;
                }               
            }
        }
    }
}
