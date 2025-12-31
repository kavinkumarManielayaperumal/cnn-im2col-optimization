#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <limits>
#include <cmath>


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

        virtual ~Layer() = default;

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) {};

        void set_input(const Tensor& input) {
            input_ = input;
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

    void read_weights_bias(std::ifstream& is) override {
    size_t wcount =
        weights_.N * weights_.C * weights_.H * weights_.W;

    size_t bcount =
        bias_.N * bias_.C * bias_.H * bias_.W;

    is.read(reinterpret_cast<char*>(weights_.data()),
            wcount * sizeof(float));

    is.read(reinterpret_cast<char*>(bias_.data()),
            bcount * sizeof(float));
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

        // Perform convolution with im2col approach 
        const size_t K=Cin*k*k;                 // kernel size flattened
        const size_t HW=Hout*Wout;             // nnumber of sliding windows

        // Reshape kernels
        Tensor Wmat(out_channels_, K, 1, 1); // (out_channels, Cin*k*k, 1, 1)
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            size_t idx = 0;
            for (size_t ic = 0; ic < Cin; ++ic) {
                for (size_t kh = 0; kh < k; ++kh) {
                    for (size_t kw = 0; kw < k; ++kw) {
                        Wmat(oc, idx++, 0, 0) = weights_(oc, ic, kh, kw);
                    }
                }
            }
        }

        // Process each batch element
        for (size_t n = 0; n < N; ++n) {

            //im2col buffer
            Tensor col(1, K, HW, 1); // (1, Cin*k*k, Hout*Wout, 1)

            size_t col_idx = 0;
            for (size_t oh = 0; oh < Hout; ++oh) {
                for (size_t ow = 0; ow < Wout; ++ow) {

                    size_t row=0;
                    for (size_t ic = 0; ic < Cin; ++ic) {
                        for (size_t kh = 0; kh < k; ++kh) {
                            for (size_t kw = 0; kw < k; ++kw) {
                                int ih = static_cast<int>(oh * stride_ + kh) - pad_;
                                int iw = static_cast<int>(ow * stride_ + kw) - pad_;

                                float val = 0.0f;
                                if (ih >= 0 && ih < static_cast<int>(Hin) &&
                                    iw >= 0 && iw < static_cast<int>(Win)) {
                                    val = input_(n, ic, ih, iw);
                                }
                                col(0, row++, col_idx, 0) = val;
                            }
                        }
                    }
                    ++col_idx;
                }
            }           

            // Matrix multiplication Wmat * col
            for (size_t oc = 0; oc < out_channels_; ++oc) {
                for (size_t hw = 0; hw < HW; ++hw) {
                    float sum =bias_(oc, 0, 0, 0); // start with bias
                    for (size_t r = 0; r < K; ++r) {
                        sum += Wmat(oc, r, 0, 0) * col(0, r, hw, 0);
                    }
                    size_t oh = hw / Wout;
                    size_t ow = hw - oh * Wout;
                    output_(n, oc, oh, ow) = sum;
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
    Linear(size_t in_features, size_t out_features)   //after convolution layer we have fully connected layer
        : Layer(LayerType::Linear),
          in_features_(in_features),
          out_features_(out_features)
          
    {
        weights_ = Tensor(out_features_, in_features_);
        bias_    = Tensor(out_features_);
    }


    void read_weights_bias(std::ifstream& is) override {
        size_t wcount =
            weights_.N * weights_.C * weights_.H * weights_.W;

        size_t bcount =
            bias_.N * bias_.C * bias_.H * bias_.W;

        is.read(reinterpret_cast<char*>(weights_.data()),
                wcount * sizeof(float));

        is.read(reinterpret_cast<char*>(bias_.data()),
                bcount * sizeof(float));
    }


    void fwd() override {
        size_t N = input_.N;
        output_ = Tensor(N, out_features_, 1, 1);

        for (size_t n = 0; n < N; ++n) {
            for (size_t o = 0; o < out_features_; ++o) {
                float sum = bias_(o, 0, 0, 0);
                for (size_t i = 0; i < in_features_; ++i) {
                    sum += input_(n, i, 0, 0) * weights_(o, i);
                }

                output_(n, o, 0, 0) = sum;
            }
        }
    }


protected:
    size_t in_features_;
    size_t out_features_;
    
};



class MaxPool2d : public Layer {
public:
    MaxPool2d(size_t kernel_size,
              size_t stride = 1,
              size_t pad = 0)
        : Layer(LayerType::MaxPool2d),
          kernel_size_(kernel_size),
          stride_(stride),
          pad_(pad) {}

    void fwd() override {
        // input dimensions
        size_t N = input_.N;
        size_t C = input_.C;
        size_t Hin = input_.H;
        size_t Win = input_.W;

        size_t k = kernel_size_;

        // output dimensions
        size_t Hout = (Hin + 2 * pad_ - k) / stride_ + 1;
        size_t Wout = (Win + 2 * pad_ - k) / stride_ + 1;

        // Output tensor
        output_ = Tensor(N, C, Hout, Wout);

        // Max pooling
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t oh = 0; oh < Hout; ++oh) {
                    for (size_t ow = 0; ow < Wout; ++ow) {

                        float max_val = -std::numeric_limits<float>::infinity();

                        for (size_t kh = 0; kh < k; ++kh) {
                            for (size_t kw = 0; kw < k; ++kw) {

                                int ih = static_cast<int>(oh * stride_ + kh) - pad_;
                                int iw = static_cast<int>(ow * stride_ + kw) - pad_;

                                if (ih >= 0 && ih < static_cast<int>(Hin) &&
                                    iw >= 0 && iw < static_cast<int>(Win)) {

                                    max_val = std::max(
                                        max_val,
                                        input_(n, c, ih, iw)
                                    );
                                }
                            }
                        }

                        output_(n, c, oh, ow) = max_val;
                    }
                }
            }
        }
    }

protected:
    size_t kernel_size_;
    size_t stride_;
    size_t pad_;
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

    void fwd() override {
        // Output has same shape as input
        output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

        for (size_t n = 0; n < input_.N; ++n) {

            // find max logit (for numerical stability)
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < input_.C; ++c) {
                max_val = std::max(max_val, input_(n, c, 0, 0));
            }

            std::cout << "[SoftMax] logits: ";
            for (size_t c = 0; c < input_.C; ++c) {
                std::cout << input_(n, c, 0, 0) << " ";
            }
            std::cout << std::endl;


            // compute exp(x - max) and sum
            float sum_exp = 0.0f;
            for (size_t c = 0; c < input_.C; ++c) {
                float e = std::exp(input_(n, c, 0, 0) - max_val);
                output_(n, c, 0, 0) = e;
                sum_exp += e;
            }

            // normalize
            for (size_t c = 0; c < input_.C; ++c) {
                output_(n, c, 0, 0) /= sum_exp;
            }
        }
    }
};



class Flatten : public Layer {
public:
    Flatten() : Layer(LayerType::Flatten) {}

    void fwd() override {
        size_t N = input_.N;
        size_t C = input_.C;
        size_t H = input_.H;
        size_t W = input_.W;

        size_t features = C * H * W;

        // Output shape: (N, features, 1, 1)
        output_ = Tensor(N, features, 1, 1);

        for (size_t n = 0; n < N; ++n) {
            size_t idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        output_(n, idx, 0, 0) = input_(n, c, h, w);
                        ++idx;
                    }
                }
            }
        }
    }
};



class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}


        ~NeuralNetwork() {
            for (auto& l : layers_) {
                delete l;
            }
        }

        void add(Layer* layer) {
            layers_.push_back(layer);
        }

        void load(std::string file) {
            std::ifstream is(file, std::ios::binary);
            if (!is.is_open()) {
                throw std::runtime_error("Failed to open file: " + file);
            }

            for (auto& layer : layers_) {
                layer->read_weights_bias(is);
            }

            is.close();
        }

        Tensor predict(Tensor input) {
            Tensor current_input = input;

            for (auto& layer : layers_) {
                layer->set_input(current_input);
                layer->fwd();
                current_input = layer->get_output();

                if (debug_) {
                    layer->print();
                }
            }

            return current_input;
        }

    private:
        bool debug_;
        std::vector<Layer*> layers_;
};

#endif // NETWORK_HPP
