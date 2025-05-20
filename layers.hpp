#include <iostream>
#include <vector>

using namespace std;

namespace Layers {
    class Dense {
        int n_regs;
        int n_inputs;
        int n_neurons;

        vector<float> weights;
        vector<float> biases;
        vector<float> inputs;
        vector<float> outputs;
        vector <float> grad;
        
        vector<float> weights_momentums;
        vector<float> weights_cache;
        vector<float> biases_momentums;
        vector<float> biases_cache;

        vector<float> dX;
        vector<float> dW;
        vector<float> dB;

        public:
            Dense(int n_inputs, int n_neurons, int n_regs) :
            n_inputs(n_inputs), n_neurons(n_neurons),
            inputs(n_regs * n_inputs), outputs(n_regs * n_neurons), grad(n_regs * n_neurons),
            weights(n_inputs * n_neurons), biases(n_neurons),
            weights_momentums(n_inputs * n_neurons), weights_cache(n_inputs * n_neurons),
            biases_momentums(n_neurons), biases_cache(n_neurons) {}

            void forward(vector<float> inputs) {
                float sum;
                for (int i = 0; i < n_regs; i++) {
                    for (int j = 0; j < n_neurons; j++) {
                        sum = 0;
                        for (int k = 0; k < n_inputs; k++) {
                            sum += inputs[i * n_inputs + k] * weights[k * n_neurons + j];
                        }
                        outputs[i * n_neurons + j] = sum + biases[j];
                    }
                }
            }

            void backward(vector<float> grad) {
                float sum;
                for (int i = 0; i < n_inputs; i++) {
                    for (int j = 0; j < n_neurons; j++) {
                        sum = 0;
                        for (int k = 0; k < n_regs; k++) {
                            sum += inputs[k * n_inputs + i] * grad[k * n_neurons + j];
                        }
                        dW[i * n_neurons + j] = sum;
                    }
                }
            }
    };
}