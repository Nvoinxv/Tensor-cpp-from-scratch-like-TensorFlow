#include "../../include/ops/inverse.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ops {

Tensor inverse(const Tensor& t) {
    auto shape = t.getShape();
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Inverse requires a square 2D matrix!");
    }
    int n = shape[0];
    Tensor out({n, n}, t.requiresGrad());

    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = t.at({i, j});
        aug[i][n + i] = 1.0;
    }

    for (int i = 0; i < n; ++i) {
        double max_el = std::abs(aug[i][i]);
        int pivot = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(aug[k][i]) > max_el) {
                max_el = std::abs(aug[k][i]);
                pivot = k;
            }
        }
        if (max_el < 1e-12) throw std::runtime_error("Matrix is singular or nearly singular!");
        if (pivot != i) std::swap(aug[i], aug[pivot]);

        double div_val = aug[i][i];
        for (int j = 0; j < 2 * n; ++j) aug[i][j] /= div_val;

        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; ++j) aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out.set({i, j}, aug[i][n + j]);
        }
    }

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t, out, n]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        if (!t.requiresGrad()) return;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double temp = 0.0;
                for (int k = 0; k < n; ++k) {
                    for (int l = 0; l < n; ++l) {
                        temp += -out.at({k, i}) * out_impl->grad[k * n + l] * out.at({j, l});
                    }
                }
                t.gradAt({i, j}) += temp;
            }
        }
    });

    return out;
}

} // namespace ops
