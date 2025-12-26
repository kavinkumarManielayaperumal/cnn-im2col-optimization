#include "network.hpp"
#include "mnist.hpp"
#include "tensor.hpp"
#include <iostream>



int main() {


    NeuralNetwork net(true);

    net.add(new Conv2d(1, 6, 5));         // Conv Layer 1
    net.add(new ReLu());                  // Activation 1
    net.add(new MaxPool2d(2, 2));         // Pooling Layer

    net.add(new Conv2d(6, 16, 5));        // Conv Layer 2
    net.add(new ReLu());                  // Activation 2
    net.add(new MaxPool2d(2, 2));         // Pooling Layer
    net.add(new Flatten());               // Flatten Layer
    net.add(new Linear(16 * 4 * 4, 120)); // Fully Connected Layer 1
    net.add(new ReLu());                  // Activation 3
    net.add(new Linear(120, 84));         // Fully Connected Layer 2
    net.add(new ReLu());                  // Activation 4
    net.add(new Linear(84, 10));          // Output Layer
    net.add(new SoftMax());               // Softmax Activation
    net.load("/home/kavin/minicnn/DATA/data-mnist-lenet.raw");

    MNIST mnist("/home/kavin/minicnn/DATA/data-mnist-t10k-images-idx3-ubyte");

    Tensor input = mnist.at(0);     //shape (1, 1, 28, 28)

    Tensor output = net.predict(input);   // shape: (1, 10, 1, 1)

    int predicted_class = 0;
    float max_prob = -1.0f;

    for (size_t c = 0; c < output.C; ++c) {
        float prob = output(0, c, 0, 0);
        std::cout << "Class " << c << ": " << prob << '\n';

        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = static_cast<int>(c);
        }
    }

    std::cout << "Predicted class: " << predicted_class << std::endl;



    return 0;
}

