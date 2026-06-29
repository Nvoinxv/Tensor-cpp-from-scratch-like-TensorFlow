#include "../../include/ops/transpose.hpp"
#include "../../include/ops/AutodiffHelper.hpp"
#include <stdexcept>

namespace ops {

Tensor transpose(const Tensor& t) {
    auto shape = t.getShape();
    if (shape.size() != 2) throw std::invalid_argument("Transpose currently supports 2D matrices!");
    int rows = shape[0], cols = shape[1];
    Tensor out({cols, rows}, t.requiresGrad());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out.set({j, i}, t.at({i, j}));
        }
    }

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t, rows, cols]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        if (!t.requiresGrad()) return;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                t.gradAt({i, j}) += out_impl->grad[j * rows + i];
            }
        }
    });
    return out;
}

} // namespace ops
