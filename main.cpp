#include <iostream>
#include <vector>

#include "layers.hpp"

using namespace std;

int main() {

    vector<Layers::Dense> layers;
    layers.push_back(Layers::Dense(1, 8, 1000));

    return 0;

}