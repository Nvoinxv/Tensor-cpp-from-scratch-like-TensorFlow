#include "../../include/ops/neg.hpp"
#include "../../include/ops/AutodiffHelper.hpp"

namespace ops {

Tensor neg(const Tensor& a) {
    Tensor out(a.getShape(), a.requiresGrad());
    const auto& da = a.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = -da[i];

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, a, [out_weak, a]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] -= og[i];
        }
    });
    return out;
}

} // namespace ops
