using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    class NeuralNetwork
    {
        private Layer[] layers;

        private int[] nrLayers;
        private Random rnd;

        public NeuralNetwork(int[] networkStruc)
        {
            layers = new Layer[networkStruc.Length];
            nrLayers = new int[networkStruc.Length];
            for(int i=0;i< networkStruc.Length-1;i++)
            {
                nrLayers[i] = networkStruc[i];
                layers[i] = new Layer(networkStruc[i], networkStruc[i + 1]);
            }
            nrLayers[networkStruc.Length - 1] = networkStruc[networkStruc.Length - 1];
        }
        private double[] Sigmoid(double[] x)
        {
            double[] outputActi = new double[x.Length];
            for(int i=0;i<x.Length;i++)
            {
                outputActi[i] = 1f / (1f + Math.Exp(-x[i]));
            }
            return outputActi;
        }
        /*
        private double SigmoidDerivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }*/
        private double[,] multiplyVectors(double[] one,double[] two)
        {
            double[,] result = new double[one.Length, two.Length];
            for(int i=0;i<one.Length;i++)
            {
                for(int j=0;j<two.Length;j++)
                {
                    result[i, j] = one[i] * two[j];
                }
            }
            return result;
        }
        private double[,] matrixTranspose(double[,] mat)
        {
            int w = mat.GetLength(0);
            int h = mat.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = mat[i, j];
                }
            }
            return result;
        }
        private double[,] multiplyMatrix(double[,] one, double[,] two)
        {
            int rA = one.GetLength(0);
            int cA = one.GetLength(1);
            int rB = two.GetLength(0);
            int cB = two.GetLength(1);
            double temp = 0;
            double[,] result = new double[rA, cB];
            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cB; j++)
                {
                    temp = 0;
                    for (int k = 0; k < cA; k++)
                    {
                        temp += one[i, k] * two[k, j];
                    }
                    result[i, j] = temp;
                }
            }
            return result;
        }
        private double[,] arrayToMatrix(double[] arr)
        {
            double[,] result = new double[arr.Length, 1];
            for (int i = 0; i < result.Length; i++)
                result[i, 0] = arr[i];
            return result;
        }
        private void ForwardPass(double[] inputs)
        {
            for(int i=0;i<layers[0].inputs.Length;i++)
            {
                layers[0].inputs[i] = inputs[i];
            }
            for(int i=0;i<layers.Length-1;i++)
            {
                layers[i].multiWeightInput();
                layers[i + 1].inputs = Sigmoid(layers[i].outputs);
            }
        }
        
        private void BackwardPass(double[][] trainInput,double[][] trainOutput)
        {
            
            double biasChange, weightChange,neuronChange;
            for(int i=0;i<trainInput.Length;i++)
            {
                ForwardPass(trainInput[i]);
                //train procedure

                //last layer 
                layers[layers.Length - 1].biasesCorrection = layers[layers.Length - 1].outputs.Select((elem, index) => elem - trainOutput[i][index]).ToArray();
                layers[layers.Length-1].weightsCorrection=multiplyVectors(layers[layers.Length - 1].biasesCorrection, layers[layers.Length-2].outputs);

                //previous layer
                layers[layers.Length-2].biasesCorrection= layers[layers.Length - 1].outputs.Select((elem, index) => elem - trainOutput[i][index]).ToArray();
                layers[layers.Length - 2].biasesCorrection = multiplyMatrix(arrayToMatrix(layers[layers.Length - 2].biasesCorrection), matrixTranspose(layers[layers.Length - 1].weights));
            }
            //apply training

           
        }
    }
}
