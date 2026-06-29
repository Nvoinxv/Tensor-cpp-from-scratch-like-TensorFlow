#include "../../include/ops/div.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <stdexcept>

namespace ops {

Tensor div(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requiresGrad() || b.requiresGrad();
    
    if (!a.isScalar() && b.isScalar()) {
        Tensor out(a.getShape(), req_grad);
        const auto& da = a.getData();
        double val_b = b.at({0});
        auto& dout = out.getMutableData();
        for (size_t i = 0; i < da.size(); ++i) dout[i] = da[i] / val_b;

        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            const auto& og = out_impl->grad;
            const auto& da = a.getData();
            double val_b = b.getData()[0];
            if (a.requiresGrad()) {
                auto& ag = a.getMutableGrad();
                for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i] / val_b;
            }
            if (b.requiresGrad()) {
                double sum_g = 0.0;
                for (size_t i = 0; i < og.size(); ++i) sum_g += og[i] * (-da[i] / (val_b * val_b));
                b.getMutableGrad()[0] += sum_g;
            }
        });
        return out;
    }

    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape mismatch in ops::div!");
    }

    Tensor out(a.getShape(), req_grad);
    const auto& da = a.getData();
    const auto& db = b.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = da[i] / db[i];

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& da = a.getData();
        const auto& db = b.getData();
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i] / db[i];
        }
        if (b.requiresGrad()) {
            auto& bg = b.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) bg[i] += og[i] * (-da[i] / (db[i] * db[i]));
        }
    });

    return out;
}

} // namespace ops
