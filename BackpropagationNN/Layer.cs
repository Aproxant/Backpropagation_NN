using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackpropagationNN
{
    class Layer
    {
        public int nrIpnut;
        public int nrOutput;

        public double[] inputs;
        public double[] outputs; //a
        public double[] sumVector; // z
        public double[,] weights; //W
        public double[,] weightsCorrection;
        public double[] biases;
        public double[] biasesCorrection;
        
        public Layer(int _in,int _out)
        {
            nrIpnut = _in;
            nrOutput = _out;

            inputs = new double[nrIpnut];
            outputs = new double[nrOutput];
            weights = new double[nrOutput, nrIpnut];
            weightsCorrection = new double[nrOutput, nrIpnut];
            biases = new double[nrOutput];
            biasesCorrection = new double[nrOutput];
            sumVector = new double[nrOutput];
        }

        public void multiWeightInput()
        {
            for(int i=0;i<weights.GetLength(0);i++)
            {
                outputs[i] = 0;
                for (int j = 0; j < weights.GetLength(1);j++)
                {                    
                    outputs[i]+= weights[i, j] * inputs[j];                        
                }
                outputs[i] += biases[i];
                sumVector[i] = outputs[i]; // sprawdzic czy nie override sumVector
                outputs[i] = Sigmoid(outputs[i]);
            }
        }
        private double Sigmoid(double x)
        {
            return 1f / (1f + Math.Exp(-x));
        }




    }
}
