#pragma once
#include "../Tensor.hpp"

namespace ops {
    Tensor dot(const Tensor& a, const Tensor& b);
    Tensor matmul(const Tensor& a, const Tensor& b);
}
