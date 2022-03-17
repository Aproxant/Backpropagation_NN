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
        private double[][] biases;
        private double[][][] weights;

        private double[][] correctedNeuron;
        private double[][] correctedBiases;
        private double[][][] correctedWeights;
        private Random rnd;

        public NeuralNetwork(int[] networkStruc)
        {
            neurons = new double[networkStruc.Length][];
            biases = new double[networkStruc.Length][];
            weights = new double[networkStruc.Length - 1][][];

            correctedNeuron = new double[networkStruc.Length][];
            correctedBiases = new double[networkStruc.Length][];
            correctedWeights = new double[networkStruc.Length - 1][][];

            for (int i=0;i<networkStruc.Length;i++)
            {
                neurons[i] = new double[networkStruc[i]];
                correctedNeuron[i] = new double[networkStruc[i]];
                biases[i] = new double[networkStruc[i]];
                correctedBiases[i] = new double[networkStruc[i]];
            }
            //InitBiases();
            InitWeights();

        }

        private void InitWeights()
        {
            for(int i=0;i<neurons.Length-1;i++)
            {
                weights[i] = new double[neurons[i].Length][];
                correctedWeights[i] = new double[neurons[i].Length][];
                for (int j=0;j<weights[i].Length;j++)
                {
                    weights[i][j] = new double[neurons[i + 1].Length];
                    correctedWeights[i][j] = new double[neurons[i + 1].Length];
                    for (int k=0;k<weights[i][j].Length;k++)
                    {
                        weights[i][j][k] = rnd.NextDouble() - 0.5f;
                    }
                }
            }
        }
        /*
        private void InitBiases()
        {
            for (int i = 0; i < neurons.Length - 1; i++)
            {
                biases[i]= rnd.NextDouble() - 0.5f;
            }
        }*/
        private double Sigmoid(double x)
        {
            return 1f / (1f + Math.Exp(-x));
        }
        private double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }
        private double[] Test(double[] input)
        {
            double sumOfLayer=0;
            for(int i=0;i<neurons[0].Length;i++)
            {
                neurons[0][i] = input[i];
            }
            
            for(int i=0;i<neurons.Length-1;i++)
            {
                for(int j=0;j<neurons[i].Length;j++)
                {
                    for(int k=0;k<neurons[i].Length;k++)
                    {
                        sumOfLayer += neurons[i][k] * weights[i][k][j];

                    }
                    neurons[i + 1][j] = Sigmoid(sumOfLayer); //+ biases[i];// add activation func before assignment done
                    correctedNeuron[i + 1][j] = neurons[i + 1][j];
                    sumOfLayer = 0;
                }               
            }
            return neurons[neurons.Length - 1];
        }
        private void Train(double[][] trainInput,double[][] trainOutput)
        {
            double biasChange, weightChange,neuronChange;
            for(int i=0;i<trainInput.Length;i++)
            {
                Test(trainInput[i]);
                for(int j=0;j<correctedNeuron[correctedNeuron.Length-1].Length;j++)
                {
                    correctedNeuron[correctedNeuron.Length - 1][j] = trainOutput[i][j];
                }
                for(int j=neurons.Length-1;j>0; j++)
                {
                    for(int k=0;k<neurons[i].Length;k++)
                    {
                        biasChange = SigmoidDerivative(neurons[j][k]) * (correctedNeuron[j][k] - neurons[j][k]);
                        correctedBiases[j][k] += biasChange;
                        for(int l=0;l<neurons[j-1].Length;l++)
                        {
                            weightChange = neurons[j - 1][l] * biasChange;
                            correctedWeights[j - 1][l][k] += weightChange;

                            neuronChange = weights[j - 1][l][k] * biasChange;
                            correctedNeuron[j - 1][l] += neuronChange;
                        }
                    }
                }                       
            }
            //apply training

            for(int i=neurons.Length-1;i>0;i++)
            {
                for(int j=0;j<neurons[i].Length;j++)
                {
                    biases[i][j] = correctedBiases[i][j];
                    correctedBiases[i][j] = 0;

                    for(int k=0;k<neurons[i-1].Length;k++)
                    {
                        weights[i - 1][k][j] += correctedWeights[i - 1][k][j];
                        correctedWeights[i - 1][k][j] = 0;
                    }
                    correctedNeuron[i][j] = 0;
                }
            }
        }
    }
}
