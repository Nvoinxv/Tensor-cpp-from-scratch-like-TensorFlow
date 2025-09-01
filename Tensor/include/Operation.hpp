#ifndef OPERATION_H
#define OPERATION_H

#include "Tensor.hpp"
#include <vector>
#include <stdexcept>

class Operation {
public:
    // Basic Operation Algebra Linear //
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor sub(const Tensor& a, const Tensor& b);
    static Tensor mul(const Tensor& a, const Tensor& b);
    static Tensor dot(const Tensor& a, const Tensor& b);

    // Activation Neuron in deep learning //
    static Tensor relu(const Tensor& t);
    static Tensor sigmoid(const Tensor& t);
    static Tensor softmax(const Tensor& t);
};

#endif
