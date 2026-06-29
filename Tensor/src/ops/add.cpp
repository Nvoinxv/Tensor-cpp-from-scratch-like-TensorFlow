#include "../../include/ops/add.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <stdexcept>

namespace ops {

Tensor add(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requiresGrad() || b.requiresGrad();
    
    if (a.isScalar() && !b.isScalar()) {
        Tensor out(b.getShape(), req_grad);
        double val_a = a.at({0});
        const auto& data_b = b.getData();
        auto& data_out = out.getMutableData();
        for (size_t i = 0; i < data_b.size(); ++i) {
            data_out[i] = val_a + data_b[i];
        }
        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            const auto& og = out_impl->grad;
            if (a.requiresGrad()) {
                double sum_g = 0.0;
                for (double g : og) sum_g += g;
                a.getMutableGrad()[0] += sum_g;
            }
            if (b.requiresGrad()) {
                auto& bg = b.getMutableGrad();
                for (size_t i = 0; i < og.size(); ++i) bg[i] += og[i];
            }
        });
        return out;
    }
    
    if (!a.isScalar() && b.isScalar()) {
        Tensor out(a.getShape(), req_grad);
        const auto& data_a = a.getData();
        double val_b = b.at({0});
        auto& data_out = out.getMutableData();
        for (size_t i = 0; i < data_a.size(); ++i) {
            data_out[i] = data_a[i] + val_b;
        }
        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            const auto& og = out_impl->grad;
            if (a.requiresGrad()) {
                auto& ag = a.getMutableGrad();
                for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i];
            }
            if (b.requiresGrad()) {
                double sum_g = 0.0;
                for (double g : og) sum_g += g;
                b.getMutableGrad()[0] += sum_g;
            }
        });
        return out;
    }

    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape mismatch in ops::add!");
    }

    Tensor out(a.getShape(), req_grad);
    const auto& da = a.getData();
    const auto& db = b.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) {
        dout[i] = da[i] + db[i];
    }

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i];
        }
        if (b.requiresGrad()) {
            auto& bg = b.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) bg[i] += og[i];
        }
    });

    return out;
}

} // namespace ops
