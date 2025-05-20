#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

namespace Activations {
    class Activation {
        public:
            virtual void forward(vector<float>& inputs) = 0;
            virtual void backward(vector<float>& grad) = 0;

            virtual ~Activation() {};
    };

    class ReLU : Activation {
        public:
            int size;
            vector<float> inputs;
            vector<float> outputs;
            vector<float> dX;

            ReLU(int n_inputs, int n_neurons) :
            size(n_inputs * n_neurons), 
            inputs(size), outputs(size), dX(size) {}

            void forward(vector<float>& inputs) override {
                this -> inputs = inputs;
                for (int i = 0; i < inputs.size(); i++) {
                    outputs[i] = max(inputs[i], 0.0f);
                }
            }

            void backward(vector<float>& grad) override {
                for (int i = 0; i < outputs.size(); i++) {
                    dX[i] = (inputs[i] <= 0) ? 0 : grad[i];
                }
            }
    };

    class Tanh : Activation {
        public:
            int size;
            vector<float> inputs;
            vector<float> outputs;
            vector<float> dX;

            Tanh(int n_inputs, int n_neurons) :
            size(n_inputs * n_neurons), 
            inputs(size), outputs(size), dX(size) {}

            void forward(vector<float>& inputs) override {
                this -> inputs = inputs;
                for (int i = 0; i < inputs.size(); i++) {
                    outputs[i] = tanhf(inputs[i]);
                }
            }

            void backward(vector<float>& grad) override {
                for (int i = 0; i < outputs.size(); i++) {
                    dX[i] = grad[i] * (1 - outputs[i] * outputs[i]);
                }
            }
    };
}