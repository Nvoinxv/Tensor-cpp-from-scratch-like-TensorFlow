#include "../../include/ops/softmax.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ops {

Tensor softmax(const Tensor& t) {
    auto shape = t.getShape();
    if (shape.empty()) throw std::invalid_argument("Softmax cannot apply to empty Tensor");

    Tensor out(shape, t.requiresGrad());
    
    if (shape.size() == 1) {
        double max_val = t.at({0});
        for (int i = 1; i < shape[0]; ++i) max_val = std::max(max_val, t.at({i}));
        
        std::vector<double> exp_vals(shape[0]);
        double sum_exp = 0.0;
        for (int i = 0; i < shape[0]; ++i) {
            exp_vals[i] = std::exp(t.at({i}) - max_val);
            sum_exp += exp_vals[i];
        }
        for (int i = 0; i < shape[0]; ++i) {
            out.set({i}, exp_vals[i] / sum_exp);
        }
    } else {
        int last_dim = shape.back();
        int outer_size = t.size() / last_dim;
        for (int outer = 0; outer < outer_size; ++outer) {
            std::vector<int> idx(shape.size(), 0);
            int rem = outer;
            for (int j = static_cast<int>(shape.size()) - 2; j >= 0; --j) {
                idx[j] = rem % shape[j];
                rem /= shape[j];
            }
            double max_val = t.at(idx);
            for (int i = 1; i < last_dim; ++i) {
                idx.back() = i;
                max_val = std::max(max_val, t.at(idx));
            }
            std::vector<double> exp_vals(last_dim);
            double sum_exp = 0.0;
            for (int i = 0; i < last_dim; ++i) {
                idx.back() = i;
                exp_vals[i] = std::exp(t.at(idx) - max_val);
                sum_exp += exp_vals[i];
            }
            for (int i = 0; i < last_dim; ++i) {
                idx.back() = i;
                out.set(idx, exp_vals[i] / sum_exp);
            }
        }
    }

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t, shape]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        if (!t.requiresGrad()) return;
        const auto& og = out_impl->grad;
        const auto& dout = out_impl->data;
        auto& tg = t.getMutableGrad();

        if (shape.size() == 1) {
            int n = shape[0];
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    double kronecker = (i == j) ? 1.0 : 0.0;
                    sum += og[j] * dout[j] * (kronecker - dout[i]);
                }
                tg[i] += sum;
            }
        } else {
            int last_dim = shape.back();
            int outer_size = t.size() / last_dim;
            for (int outer = 0; outer < outer_size; ++outer) {
                int offset = outer * last_dim;
                for (int i = 0; i < last_dim; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < last_dim; ++j) {
                        double kronecker = (i == j) ? 1.0 : 0.0;
                        sum += og[offset + j] * dout[offset + j] * (kronecker - dout[offset + i]);
                    }
                    tg[offset + i] += sum;
                }
            }
        }
    });

    return out;
}

} // namespace ops
