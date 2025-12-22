#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}

class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void set_input(const Tensor& input) {
            input_ = input;
        }

        }
        Tensor get_output() const {
            return output_;
        }


        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
public:
    Conv2d(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride = 1,
           size_t pad = 0)        // this layer's constructor, so we can set specific attributes for different layer types
        : Layer(LayerType::Conv2d),
          in_channels_(in_channels),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          stride_(stride),
          pad_(pad)
    {
        weights_ = Tensor(out_channels_, in_channels_, kernel_size_, kernel_size_);
        bias_    = Tensor(out_channels_);
    }
    // override the forward method
    void fwd() override {
        size_t N = input_.N;
        size_t Cin = input_.C;
        size_t Hin = input_.H;
        size_t Win = input_.W;

        size_t k = kernel_size_;

        size_t Hout = (Hin + 2 * pad_ - k) / stride_ + 1;
        size_t Wout = (Win + 2 * pad_ - k) / stride_ + 1;

        output_ = Tensor(N, out_channels_, Hout, Wout);

        // Perform convolution
        for (size_t n = 0; n < N; ++n) {
            for (size_t oc = 0; oc < out_channels_; ++oc) {
                for (size_t oh = 0; oh < Hout; ++oh) {
                    for (size_t ow = 0; ow < Wout; ++ow) {

                        float sum = bias_(oc);

                        for (size_t ic = 0; ic < Cin; ++ic) {
                            for (size_t kh = 0; kh < k; ++kh) {
                                for (size_t kw = 0; kw < k; ++kw) {

                                    int ih = oh * stride_ + kh - pad_;
                                    int iw = ow * stride_ + kw - pad_;

                                    if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                                        sum += input_(n, ic, ih, iw) * weights_(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                        output_(n, oc, oh, ow) = sum;
                    }
                }
            }
        }
    }


protected:  // expect derived classes to access these members
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
};




class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear) {}
    // TODO
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d) {}
    // TODO
};


class ReLu : public Layer {
public:
    ReLu() : Layer(LayerType::ReLu) {}

    void fwd() override {
        // Output has same shape as input
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

        for (size_t n = 0; n < input_.N; ++n) {
            for (size_t c = 0; c < input_.C; ++c) {
                for (size_t h = 0; h < input_.H; ++h) {
                    for (size_t w = 0; w < input_.W; ++w) {
                        output_(n, c, h, w) =
                            std::max(0.0f, input_(n, c, h, w));
                    }
                }
            }
        }
    }
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    // TODO
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    // TODO
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            // TODO
        }

        void load(std::string file) {
            // TODO
        }

        Tensor predict(Tensor input) {
            // TODO
        }

    private:
        bool debug_;
        // TODO: storage for layers
};

#endif // NETWORK_HPP
