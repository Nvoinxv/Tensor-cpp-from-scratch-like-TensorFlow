#include "../../include/ops/mean.hpp"
#include "../../include/ops/AutodiffHelper.hpp"

namespace ops {

Tensor mean(const Tensor& t) {
    bool req_grad = t.requiresGrad();
    double s = 0.0;
    const auto& dt = t.getData();
    for (double val : dt) s += val;
    double N = static_cast<double>(dt.size());
    Tensor out({1}, {s / (N > 0 ? N : 1.0)}, req_grad);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t, N]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        if (!t.requiresGrad()) return;
        double og = out_impl->grad[0] / (N > 0 ? N : 1.0);
        auto& tg = t.getMutableGrad();
        for (size_t i = 0; i < tg.size(); ++i) tg[i] += og;
    });
    return out;
}

} // namespace ops
