#include "../include/Operation.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <numeric>

// ==========================================
// Helper functions for autodiff closures
// ==========================================

static void attach_binary_backward(Tensor& out, const Tensor& a, const Tensor& b, std::function<void()> bwd) {
    if (!out.requiresGrad()) return;
    out.getImpl()->parents.push_back(a);
    out.getImpl()->parents.push_back(b);
    out.getImpl()->backward_fn = bwd;
}

static void attach_unary_backward(Tensor& out, const Tensor& a, std::function<void()> bwd) {
    if (!out.requiresGrad()) return;
    out.getImpl()->parents.push_back(a);
    out.getImpl()->backward_fn = bwd;
}

// ==========================================
// Basic Algebra Operations
// ==========================================

Tensor Operation::add(const Tensor& a, const Tensor& b) {
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
        throw std::invalid_argument("Shape mismatch in Operation::add!");
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

Tensor Operation::sub(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requiresGrad() || b.requiresGrad();
    
    if (a.isScalar() && !b.isScalar()) {
        Tensor out(b.getShape(), req_grad);
        double val_a = a.at({0});
        const auto& data_b = b.getData();
        auto& data_out = out.getMutableData();
        for (size_t i = 0; i < data_b.size(); ++i) {
            data_out[i] = val_a - data_b[i];
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
                for (size_t i = 0; i < og.size(); ++i) bg[i] -= og[i];
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
            data_out[i] = data_a[i] - val_b;
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
                b.getMutableGrad()[0] -= sum_g;
            }
        });
        return out;
    }

    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape mismatch in Operation::sub!");
    }

    Tensor out(a.getShape(), req_grad);
    const auto& da = a.getData();
    const auto& db = b.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) {
        dout[i] = da[i] - db[i];
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
            for (size_t i = 0; i < og.size(); ++i) bg[i] -= og[i];
        }
    });

    return out;
}

Tensor Operation::mul(const Tensor& a, const Tensor& b) {
    bool req_grad = a.requiresGrad() || b.requiresGrad();
    
    if (a.isScalar() && !b.isScalar()) {
        Tensor out(b.getShape(), req_grad);
        double val_a = a.at({0});
        const auto& db = b.getData();
        auto& dout = out.getMutableData();
        for (size_t i = 0; i < db.size(); ++i) dout[i] = val_a * db[i];
        
        auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
        attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
            auto out_impl = out_weak.lock(); if (!out_impl) return;
            const auto& og = out_impl->grad;
            const auto& db = b.getData();
            if (a.requiresGrad()) {
                double sum_g = 0.0;
                for (size_t i = 0; i < og.size(); ++i) sum_g += og[i] * db[i];
                a.getMutableGrad()[0] += sum_g;
            }
            if (b.requiresGrad()) {
                double val_a = a.getData()[0];
                auto& bg = b.getMutableGrad();
                for (size_t i = 0; i < og.size(); ++i) bg[i] += og[i] * val_a;
            }
        });
        return out;
    }
    
    if (!a.isScalar() && b.isScalar()) {
        return Operation::mul(b, a);
    }

    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape mismatch in Operation::mul!");
    }

    Tensor out(a.getShape(), req_grad);
    const auto& da = a.getData();
    const auto& db = b.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = da[i] * db[i];

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_binary_backward(out, a, b, [out_weak, a, b]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& da = a.getData();
        const auto& db = b.getData();
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i] * db[i];
        }
        if (b.requiresGrad()) {
            auto& bg = b.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) bg[i] += og[i] * da[i];
        }
    });

    return out;
}

Tensor Operation::div(const Tensor& a, const Tensor& b) {
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
        throw std::invalid_argument("Shape mismatch in Operation::div!");
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

Tensor Operation::neg(const Tensor& a) {
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

Tensor Operation::pow(const Tensor& a, double exponent) {
    Tensor out(a.getShape(), a.requiresGrad());
    const auto& da = a.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = std::pow(da[i], exponent);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, a, [out_weak, a, exponent]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& da = a.getData();
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) {
                ag[i] += og[i] * exponent * std::pow(da[i], exponent - 1.0);
            }
        }
    });
    return out;
}

Tensor Operation::exp(const Tensor& a) {
    Tensor out(a.getShape(), a.requiresGrad());
    const auto& da = a.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = std::exp(da[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, a, [out_weak, a]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dout = out_impl->data;
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i] * dout[i];
        }
    });
    return out;
}

Tensor Operation::log(const Tensor& a) {
    Tensor out(a.getShape(), a.requiresGrad());
    const auto& da = a.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < da.size(); ++i) dout[i] = std::log(da[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, a, [out_weak, a]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& da = a.getData();
        if (a.requiresGrad()) {
            auto& ag = a.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) ag[i] += og[i] / da[i];
        }
    });
    return out;
}

// ==========================================
// Trigonometric & Hyperbolic Operations
// ==========================================

Tensor Operation::sin(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = std::sin(dt[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dt = t.getData();
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) tg[i] += og[i] * std::cos(dt[i]);
        }
    });
    return out;
}

Tensor Operation::cos(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = std::cos(dt[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dt = t.getData();
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) tg[i] += og[i] * (-std::sin(dt[i]));
        }
    });
    return out;
}

Tensor Operation::tan(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = std::tan(dt[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dout = out_impl->data;
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) tg[i] += og[i] * (1.0 + dout[i] * dout[i]);
        }
    });
    return out;
}

Tensor Operation::tanh(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = std::tanh(dt[i]);

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dout = out_impl->data;
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) tg[i] += og[i] * (1.0 - dout[i] * dout[i]);
        }
    });
    return out;
}

// ==========================================
// Activation Functions
// ==========================================

Tensor Operation::relu(const Tensor& t) {
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

Tensor Operation::sigmoid(const Tensor& t) {
    Tensor out(t.getShape(), t.requiresGrad());
    const auto& dt = t.getData();
    auto& dout = out.getMutableData();
    for (size_t i = 0; i < dt.size(); ++i) dout[i] = 1.0 / (1.0 + std::exp(-dt[i]));

    auto out_weak = std::weak_ptr<TensorImpl>(out.getImpl());
    attach_unary_backward(out, t, [out_weak, t]() mutable {
        auto out_impl = out_weak.lock(); if (!out_impl) return;
        const auto& og = out_impl->grad;
        const auto& dout = out_impl->data;
        if (t.requiresGrad()) {
            auto& tg = t.getMutableGrad();
            for (size_t i = 0; i < og.size(); ++i) tg[i] += og[i] * dout[i] * (1.0 - dout[i]);
        }
    });
    return out;
}

Tensor Operation::softmax(const Tensor& t) {
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

// ==========================================
// Linear Algebra & Reductions
// ==========================================

Tensor Operation::dot(const Tensor& a, const Tensor& b) {
    return Operation::matmul(a, b);
}

Tensor Operation::matmul(const Tensor& a, const Tensor& b) {
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
            // dA = dC * B^T
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
            // dB = A^T * dC
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

Tensor Operation::transpose(const Tensor& t) {
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

Tensor Operation::inverse(const Tensor& t) {
    auto shape = t.getShape();
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Inverse requires a square 2D matrix!");
    }
    int n = shape[0];
    Tensor out({n, n}, t.requiresGrad());

    // Augmented matrix [A | I]
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = t.at({i, j});
        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination
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
        // dX = - Y^T * dY * Y^T where Y = out
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double temp = 0.0;
                for (int k = 0; k < n; ++k) {
                    for (int l = 0; l < n; ++l) {
                        // -Y^T(i, k) * dY(k, l) * Y^T(l, j) = -Y(k, i) * dY(k, l) * Y(j, l)
                        temp += -out.at({k, i}) * out_impl->grad[k * n + l] * out.at({j, l});
                    }
                }
                t.gradAt({i, j}) += temp;
            }
        }
    });

    return out;
}

Tensor Operation::sum(const Tensor& t) {
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

Tensor Operation::mean(const Tensor& t) {
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
