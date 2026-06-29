#include "../../include/ops/pow.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <cmath>

namespace ops {

Tensor pow(const Tensor& a, double exponent) {
    Tensor out(a.getShape(), a.requiresGrad());
    const auto& da = a.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = std::pow(da[i], exponent);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, a, [out_weak, a, exponent]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& da = a.getData();
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) {
                ag[i] += og[i] * exponent * std::pow(da[i], exponent - 1.0);
            }
        }
    });
    return out;
}

} // namespace ops
