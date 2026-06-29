#pragma once
#include "../Tensor.hpp"
#include <functional>

namespace ops {

inline void attach_binary_backward(Tensor& out, const Tensor& a, const Tensor& b, std::function<void()> bwd) {
    if (!out.requiresGrad()) return;
    out.getImpl()->parents.push_back(a);
    out.getImpl()->parents.push_back(b);
    out.getImpl()->backward_fn = bwd;
}

inline void attach_unary_backward(Tensor& out, const Tensor& a, std::function<void()> bwd) {
    if (!out.requiresGrad()) return;
    out.getImpl()->parents.push_back(a);
    out.getImpl()->backward_fn = bwd;
}

} // namespace ops
