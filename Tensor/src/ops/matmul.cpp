#include "../../include/ops/matmul.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <stdexcept>

namespace ops {

Tensor dot(const Tensor& a, const Tensor& b) {
    return ops::matmul(a, b);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    auto shapeA = a.getShape();
    auto shapeB = b.getShape();

    if (shapeA.size() == 1 && shapeB.size() == 1) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector dot mismatch!");
        bool req_grad = a.requiresGrad() || b.requiresGrad();
        double sum = 0.0;
        for (int i = 0; i < a.size(); ++i) sum += a.at({i}) * b.at({i});
        Tensor out({1}, {sum}, req_grad);

        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            double og = out_impl->grad[0];
            if (a.requiresGrad()) {
                auto& ag = a.getMutableGrad();
                const auto& db = b.getData();
                for (size_t i = 0; i < ag.size(); ++i) ag[i] += og * db[i];
            }
            if (b.requiresGrad()) {
                auto& bg = b.getMutableGrad();
                const auto& da = a.getData();
                for (size_t i = 0; i < bg.size(); ++i) bg[i] += og * da[i];
            }
        });
        return out;
    }

    if (shapeA.size() == 2 && shapeB.size() == 2) {
        if (shapeA[1] != shapeB[0]) throw std::invalid_argument("2D Matmul dimension mismatch!");
        int m = shapeA[0], n = shapeA[1], p = shapeB[1];
        bool req_grad = a.requiresGrad() || b.requiresGrad();
        Tensor out({m, p}, req_grad);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += a.at({i, k}) * b.at({k, j});
                }
                out.set({i, j}, sum);
            }
        }

        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b, m, n, p]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            if (a.requiresGrad()) {
                for (int i = 0; i < m; ++i) {
                    for (int k = 0; k < n; ++k) {
                        double sum = 0.0;
                        for (int j = 0; j < p; ++j) {
                            sum += out_impl->grad[i * p + j] * b.at({k, j});
                        }
                        a.gradAt({i, k}) += sum;
                    }
                }
            }
            if (b.requiresGrad()) {
                for (int k = 0; k < n; ++k) {
                    for (int j = 0; j < p; ++j) {
                        double sum = 0.0;
                        for (int i = 0; i < m; ++i) {
                            sum += a.at({i, k}) * out_impl->grad[i * p + j];
                        }
                        b.gradAt({k, j}) += sum;
                    }
                }
            }
        });
        return out;
    }

    throw std::invalid_argument("Matmul currently supports 1D vectors and 2D matrices!");
}

} // namespace ops
