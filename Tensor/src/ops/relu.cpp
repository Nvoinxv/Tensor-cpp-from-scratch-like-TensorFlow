#include "../../include/ops/relu.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <algorithm>

namespace ops {

Tensor relu(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = std::max(0.0, dt[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dt = t.getData();
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) {
                if (dt[i] > 0.0) tg[i] += og[i];
            }
        }
    });
    return out;
}

} // namespace ops
