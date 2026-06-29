#include "../include/Tensor.hpp"
#include "../include/ops/all_ops.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include <unordered_set>

// ==========================================
// TensorImpl Methods
// ==========================================

void TensorImpl::computeStrides() {
    strides.resize(shape.size());
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

int TensorImpl::computeTotalSize(const std::vector<int>& shp) const {
    if (shp.empty()) return 0;
    for (int dim : shp) {
        if (dim <= 0) return 0;
    }
    return std::accumulate(shp.begin(), shp.end(), 1, std::multiplies<int>());
}

int TensorImpl::flattenIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("The index number does not match!");
    }
    int flatIndex = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimensional " + std::to_string(i));
        }
        flatIndex += indices[i] * strides[i];
    }
    return flatIndex;
}

TensorImpl::TensorImpl(const std::vector<int>& shape, bool req_grad)
    : shape(shape), total_size(computeTotalSize(shape)), requires_grad(req_grad) {
    computeStrides();
    data.resize(total_size, 0.0);
    grad.resize(total_size, 0.0);
}

TensorImpl::TensorImpl(const std::vector<int>& shape, const std::vector<double>& values, bool req_grad)
    : shape(shape), total_size(computeTotalSize(shape)), requires_grad(req_grad) {
    if (values.size() != static_cast<size_t>(total_size)) {
        throw std::invalid_argument("The index number does not match!");
    }
    computeStrides();
    data = values;
    grad.resize(total_size, 0.0);
}

// ==========================================
// Tensor Constructors & Factory Methods
// ==========================================

Tensor::Tensor() : impl(nullptr) {}

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad)
    : impl(std::make_shared<TensorImpl>(shape, requires_grad)) {}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& values, bool requires_grad)
    : impl(std::make_shared<TensorImpl>(shape, values, requires_grad)) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> ptr) : impl(ptr) {}

Tensor Tensor::zeros(const std::vector<int>& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);
}

Tensor Tensor::ones(const std::vector<int>& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.getMutableData().begin(), t.getMutableData().end(), 1.0);
    return t;
}

Tensor Tensor::randn(const std::vector<int>& shape, double mean, double stddev, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(mean, stddev);
    auto& mutable_data = t.getMutableData();
    for (double& val : mutable_data) {
        val = d(gen);
    }
    return t;
}

// ==========================================
// Element Accessors
// ==========================================

double& Tensor::operator()(const std::initializer_list<int>& indices) {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data[impl->flattenIndex(indices)];
}

const double& Tensor::operator()(const std::initializer_list<int>& indices) const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data[impl->flattenIndex(indices)];
}

double Tensor::at(const std::vector<int>& indices) const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data[impl->flattenIndex(indices)];
}

double& Tensor::at(const std::vector<int>& indices) {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data[impl->flattenIndex(indices)];
}

void Tensor::set(const std::vector<int>& indices, double value) {
    at(indices) = value;
}

void Tensor::apply(const std::function<double(double)>& func) {
    if (!impl) return;
    for (double& val : impl->data) {
        val = func(val);
    }
}

// ==========================================
// Getters & Metadata
// ==========================================

std::vector<int> Tensor::getShape() const { return impl ? impl->shape : std::vector<int>{}; }
int Tensor::size() const { return impl ? impl->total_size : 0; }
int Tensor::rank() const { return impl ? static_cast<int>(impl->shape.size()) : 0; }
bool Tensor::isScalar() const { return impl && (impl->shape.empty() || (impl->shape.size() == 1 && impl->shape[0] == 1)); }
bool Tensor::isEmpty() const { return !impl || impl->total_size == 0; }

const std::vector<double>& Tensor::getData() const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data;
}

std::vector<double>& Tensor::getMutableData() const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->data;
}

const std::vector<int>& Tensor::getStrides() const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->strides;
}

// ==========================================
// Autodiff / Gradient Methods
// ==========================================

bool Tensor::requiresGrad() const { return impl ? impl->requires_grad : false; }

void Tensor::setRequiresGrad(bool req) {
    if (impl) impl->requires_grad = req;
}

const std::vector<double>& Tensor::getGrad() const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->grad;
}

std::vector<double>& Tensor::getMutableGrad() const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->grad;
}

double& Tensor::gradAt(const std::vector<int>& indices) const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    return impl->grad[impl->flattenIndex(indices)];
}

void Tensor::zero_grad() {
    if (!impl) return;
    std::fill(impl->grad.begin(), impl->grad.end(), 0.0);
}

void Tensor::backward() {
    if (!impl || !impl->requires_grad) return;

    // Check if initial loss gradient is zero, seed with 1.0
    bool all_zero = true;
    for (double g : impl->grad) {
        if (g != 0.0) { all_zero = false; break; }
    }
    if (all_zero) {
        std::fill(impl->grad.begin(), impl->grad.end(), 1.0);
    }

    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::unordered_set<TensorImpl*> visited;

    std::function<void(std::shared_ptr<TensorImpl>)> build_topo = [&](std::shared_ptr<TensorImpl> node) {
        if (!node || visited.find(node.get()) != visited.end()) return;
        visited.insert(node.get());
        for (auto& parent : node->parents) {
            if (parent.getImpl()) {
                build_topo(parent.getImpl());
            }
        }
        topo.push_back(node);
    };

    build_topo(impl);

    // Run backward closures in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            ((*it)->backward_fn)();
        }
    }
}

// ==========================================
// Operations & Manipulation
// ==========================================

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    int new_total_size = impl->computeTotalSize(new_shape);
    if (new_total_size != impl->total_size) {
        throw std::invalid_argument("Reshape: size mismatch!");
    }
    Tensor res(new_shape, impl->data, impl->requires_grad);
    res.getMutableGrad() = impl->grad;
    return res;
}

Tensor Tensor::slice(const std::vector<std::pair<int, int>>& ranges) const {
    if (!impl) throw std::runtime_error("Uninitialized Tensor");
    if (ranges.size() != impl->shape.size()) {
        throw std::invalid_argument("Rank mismatch for slice!");
    }

    std::vector<int> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        int start = ranges[i].first;
        int end = ranges[i].second;
        if (start < 0) start += impl->shape[i];
        if (end < 0) end += impl->shape[i];
        if (start >= end || start < 0 || end > impl->shape[i]) {
            throw std::out_of_range("Invalid slice range for dimension " + std::to_string(i));
        }
        new_shape.push_back(end - start);
    }

    Tensor result(new_shape);

    std::function<void(std::vector<int>&, std::vector<int>&, int)> slice_rec;
    slice_rec = [&](std::vector<int>& src_idx, std::vector<int>& dst_idx, int dim) {
        if (dim == static_cast<int>(impl->shape.size())) {
            result.at(dst_idx) = this->at(src_idx);
            return;
        }
        int start = ranges[dim].first;
        if (start < 0) start += impl->shape[dim];
        for (int i = 0; i < new_shape[dim]; ++i) {
            src_idx[dim] = start + i;
            dst_idx[dim] = i;
            slice_rec(src_idx, dst_idx, dim + 1);
        }
    };

    std::vector<int> src_idx(impl->shape.size());
    std::vector<int> dst_idx(new_shape.size());
    slice_rec(src_idx, dst_idx, 0);

    return result;
}

// Operator Overloads
Tensor Tensor::operator+(const Tensor& other) const { return ops::add(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return ops::sub(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return ops::mul(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return ops::div(*this, other); }
Tensor Tensor::operator-() const { return ops::neg(*this); }

Tensor Tensor::operator+(double val) const { return ops::add(*this, Tensor({1}, std::vector<double>{val})); }
Tensor Tensor::operator-(double val) const { return ops::sub(*this, Tensor({1}, std::vector<double>{val})); }
Tensor Tensor::operator*(double val) const { return ops::mul(*this, Tensor({1}, std::vector<double>{val})); }
Tensor Tensor::operator/(double val) const { return ops::div(*this, Tensor({1}, std::vector<double>{val})); }

// ==========================================
// Formatting & Printing
// ==========================================

void Tensor::printRecursive(const std::vector<int>& indices, int dim) const {
    if (dim == static_cast<int>(impl->shape.size()) - 1) {
        std::cout << "[";
        for (int i = 0; i < impl->shape[dim]; i++) {
            auto idx = indices;
            idx.push_back(i);
            std::cout << impl->data[impl->flattenIndex(idx)];
            if (i < impl->shape[dim] - 1) std::cout << ", ";
        }
        std::cout << "]";
    } else {
        std::cout << "[";
        for (int i = 0; i < impl->shape[dim]; i++) {
            if (i > 0) {
                std::cout << ",\n";
                for (int j = 0; j <= dim; j++) std::cout << " ";
            }
            auto idx = indices;
            idx.push_back(i);
            printRecursive(idx, dim + 1);
        }
        std::cout << "]";
    }
}

void Tensor::print() const {
    if (!impl || impl->total_size == 0) {
        std::cout << "Empty Tensor\n";
        return;
    }

    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < impl->shape.size(); ++i) {
        std::cout << impl->shape[i];
        if (i < impl->shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], requires_grad=" << (impl->requires_grad ? "true" : "false") << ",\ndata=";
    printRecursive({}, 0);

    if (impl->requires_grad) {
        std::cout << ",\ngrad=[";
        for (size_t i = 0; i < impl->grad.size(); ++i) {
            std::cout << impl->grad[i];
            if (i < impl->grad.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
    std::cout << ")\n";
}

void Tensor::reserve(size_t capacity) {
    if (impl) {
        impl->data.reserve(capacity);
        impl->grad.reserve(capacity);
    }
}

void Tensor::shrink_to_fit() {
    if (impl) {
        impl->data.shrink_to_fit();
        impl->grad.shrink_to_fit();
    }
}
