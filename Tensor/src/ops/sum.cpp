#include "../../include/ops/sum.hpp"
#include "../../include/ops/AutodiffHelper.hpp"

namespace ops {

Tensor sum(const Tensor& t) {
    bool req_grad = t.requiresGrad();
    double s = 0.0;
    const auto& dt = t.getData();
    for (double val : dt) s += val;
    Tensor out({1}, {s}, req_grad);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        if (!t.requiresGrad()) return;
        double og = out_impl->grad[0];
        auto& tg = t.getMutableGrad();
        for (size_t i = 0; i < tg.size(); ++i) tg[i] += og;
    });
    return out;
}

} // namespace ops
