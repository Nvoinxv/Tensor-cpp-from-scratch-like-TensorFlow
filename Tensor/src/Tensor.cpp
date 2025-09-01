#include "../include/Tensor.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>

// Tensor.cpp is all logic about workflow Tensor //
// Precompute strides for optimization all element //
void Tensor::computeStrides() {
    strides.resize(shape.size());
    if (!shape.empty()) {
        strides[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}


// Calculate all shape (optimized with early termination) //
int Tensor::computeTotalSize(const std::vector<int>& shape) const {
    if (shape.empty()) return 0;
    
    // Check for zero dimensions //
    for (int dim : shape) {
        if (dim <= 0) return 0;
    }
    
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// Flatten index with precomputed strides //
int Tensor::flattenIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("The index number does not match!");
    }
    
    int flatIndex = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        // Bounds checking //
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimensional " + std::to_string(i));
        }
        flatIndex += indices[i] * strides[i];
    }
    return flatIndex;
}

// Constructor kosong dengan optimized memory allocation //
Tensor::Tensor(const std::vector<int>& shape) 
    : shape(shape), total_size(computeTotalSize(shape)) {
    computeStrides();
    
    // Reserve memory untuk menghindari realokasi
    data.reserve(total_size);
    data.resize(total_size, 0.0);
}

// Constructor with data //
Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& values)
    : shape(shape), total_size(computeTotalSize(shape)) {
    if (values.size() != static_cast<size_t>(total_size)) {
        throw std::invalid_argument("The index number does not match!");
    }
    
    computeStrides();
    
    // Efficient copy dengan reserve //
    data.reserve(total_size);
    data = values;
}

// Optimized operator accses with inline computation //
double& Tensor::operator()(const std::initializer_list<int>& indices) {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("The index number does not match!");
    }
    
    int flatIndex = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < shape.size(); ++i, ++it) {
        if (*it < 0 || *it >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimensional " + std::to_string(i));
        }
        flatIndex += (*it) * strides[i];
    }
    
    return data[flatIndex];
}

// Const version with optimization //
const double& Tensor::operator()(const std::initializer_list<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("The index number does not match!");
    }
    
    int flatIndex = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < shape.size(); ++i, ++it) {
        if (*it < 0 || *it >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimensional " + std::to_string(i));
        }
        flatIndex += (*it) * strides[i];
    }
    
    return data[flatIndex];
}

// Getter which efficient //
std::vector<int> Tensor::getShape() const { 
    return shape; 
}

int Tensor::size() const { 
    return total_size; 
}

// Optimized at function with precomputed strides //
double Tensor::at(const std::vector<int>& indices) const {
    return data[flattenIndex(indices)];
}

double& Tensor::at(const std::vector<int>& indices) {
    return data[flattenIndex(indices)];
}

// Setter which use optimized at //
void Tensor::set(const std::vector<int>& indices, double value) {
    at(indices) = value;
}


// Optimized print with better formatting //
void Tensor::printRecursive(const std::vector<int>& indices, int dim) const {
    if (dim == static_cast<int>(shape.size()) - 1) {
        std::cout << "[";
        for (int i = 0; i < shape[dim]; i++) {
            auto idx = indices;
            idx.push_back(i);
            
            // Use optimized flattenIndex
            std::cout << data[flattenIndex(idx)];
            if (i < shape[dim] - 1) std::cout << ", ";
        }
        std::cout << "]";
    } else {
        std::cout << "[";
        for (int i = 0; i < shape[dim]; i++) {
            if (i > 0) {
                std::cout << ",\n";
                // Indentation untuk readability
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
    if (total_size == 0) {
        std::cout << "Empty Tensor with shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        return;
    }
    
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=";
    
    printRecursive({}, 0);
    std::cout << ")\n";
}

// Additional methods for utility //
bool Tensor::isScalar() const {
    return shape.empty() || (shape.size() == 1 && shape[0] == 1);
}

bool Tensor::isEmpty() const {
    return total_size == 0;
}

int Tensor::rank() const {
    return static_cast<int>(shape.size());
}

// Memory efficient reshaping //
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_total_size = computeTotalSize(new_shape);
    if (new_total_size != total_size) {
        throw std::invalid_argument("Reshape: size it's not sweatble!");
    }
    
    return Tensor(new_shape, data);
}

// Efficient slicing operation //
Tensor Tensor::slice(const std::vector<std::pair<int, int>>& ranges) const {
    if (ranges.size() != shape.size()) {
        throw std::invalid_argument("The number rank Tensor must the same!");
    }
    
    std::vector<int> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        int start = ranges[i].first;
        int end = ranges[i].second;
        if (start < 0) start += shape[i];
        if (end < 0) end += shape[i];
        if (start >= end || start < 0 || end > shape[i]) {
            throw std::out_of_range("Invalid slice range for dimensional " + std::to_string(i));
        }
        new_shape.push_back(end - start);
    }
    
    Tensor result(new_shape);
    
    // Recursive slicing (simplified implementation)
    std::function<void(std::vector<int>&, std::vector<int>&, int)> slice_recursive;
    slice_recursive = [&](std::vector<int>& src_idx, std::vector<int>& dst_idx, int dim) {
        if (dim == static_cast<int>(shape.size())) {
            result.at(dst_idx) = this->at(src_idx);
            return;
        }
        
        int start = ranges[dim].first;
        if (start < 0) start += shape[dim];
        
        for (int i = 0; i < new_shape[dim]; ++i) {
            src_idx[dim] = start + i;
            dst_idx[dim] = i;
            slice_recursive(src_idx, dst_idx, dim + 1);
        }
    };
    
    std::vector<int> src_idx(shape.size());
    std::vector<int> dst_idx(new_shape.size());
    slice_recursive(src_idx, dst_idx, 0);
    
    return result;
}