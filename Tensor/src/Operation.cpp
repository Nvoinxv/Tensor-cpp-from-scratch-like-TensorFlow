#include "../include/Operation.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>

// Helper function for accses tensor data with flat index //
double getTensorValue(const Tensor& t, int flatIndex) {
    auto shape = t.getShape();
    std::vector<int> indices(shape.size());
    
    int remaining = flatIndex;
    for (int i = shape.size() - 1; i >= 0; --i) {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    
    return t.at(indices);
}

// I make a logic function for all operation and activation neuron deep learning //
// Because this is a test for Tensor //
// I use Tensor calculate not a vector libary c++ //

// Addition tensor //
Tensor Operation::add(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape is not same!");
    }
    
    auto shape = a.getShape();
    Tensor result(shape);
    
    // Iterasi melalui semua elemen dengan flat index //
    for (int i = 0; i < a.size(); ++i) {
        // Convert flat index ke multi-dimensional indices //
        std::vector<int> indices(shape.size());
        int remaining = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        double val_a = a.at(indices);
        double val_b = b.at(indices);
        result.set(indices, val_a + val_b);
    }
    
    return result;
}

// Subtraction tensor //
Tensor Operation::sub(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape it's not sweetable for operetion sub!");
    }
    
    auto shape = a.getShape();
    Tensor result(shape);
    
    for (int i = 0; i < a.size(); ++i) {
        std::vector<int> indices(shape.size());
        int remaining = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        double val_a = a.at(indices);
        double val_b = b.at(indices);
        result.set(indices, val_a - val_b);
    }
    
    return result;
}

// Multiple element-wise //
Tensor Operation::mul(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("Shape it's not sweetable for operetion mul!");
    }
    
    auto shape = a.getShape();
    Tensor result(shape);
    
    for (int i = 0; i < a.size(); ++i) {
        std::vector<int> indices(shape.size());
        int remaining = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        double val_a = a.at(indices);
        double val_b = b.at(indices);
        result.set(indices, val_a * val_b);
    }
    
    return result;
}

// Dot product //
Tensor Operation::dot(const Tensor& a, const Tensor& b) {
    auto shapeA = a.getShape();
    auto shapeB = b.getShape();

    // Case 1: Vector dot (1D) //
    if (shapeA.size() == 1 && shapeB.size() == 1) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Dimensional not sweetable untuk dot 1D!");
        }
        double sum = 0.0;
        for (int i = 0; i < a.size(); ++i) {
            double val_a = a.at({i});
            double val_b = b.at({i});
            sum += val_a * val_b;
        }
        return Tensor({1}, {sum});
    }

    // Case 2: Matrix multiplication (2D) //
    if (shapeA.size() == 2 && shapeB.size() == 2) {
        if (shapeA[1] != shapeB[0]) {
            throw std::invalid_argument("Dimensional not sweetable for matrix dot!");
        }
        
        int m = shapeA[0], n = shapeA[1], p = shapeB[1];
        Tensor result({m, p});

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    double val_a = a.at({i, k});
                    double val_b = b.at({k, j});
                    sum += val_a * val_b;
                }
                result.set({i, j}, sum);
            }
        }
        return result;
    }

    // Case 3: Batched matmul (>2D) //
    if (shapeA.size() > 2 && shapeB.size() > 2) {
        if (shapeA.size() != shapeB.size()) {
            throw std::invalid_argument("Batched matmul need same rank tensor!");
        }
        
        // Validasi batch dimensions
        for (int i = 0; i < shapeA.size() - 2; ++i) {
            if (shapeA[i] != shapeB[i]) {
                throw std::invalid_argument("Batch dimensional it's not sweetable!");
            }
        }

        int m = shapeA[shapeA.size()-2];
        int kA = shapeA[shapeA.size()-1];
        int kB = shapeB[shapeB.size()-2];
        int n = shapeB[shapeB.size()-1];

        if (kA != kB) {
            throw std::invalid_argument("Inner dimensional it's not sweetable!");
        }

        std::vector<int> outShape = shapeA;
        outShape[outShape.size()-2] = m;
        outShape[outShape.size()-1] = n;

        Tensor result(outShape);

        // Generate all combiancation batch and indicies //
        std::function<void(std::vector<int>&, int)> processBatch;
        processBatch = [&](std::vector<int>& batchIdx, int dim) {
            if (dim == shapeA.size() - 2) {
                // Process matrix multiplication for this batch //
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < n; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < kA; ++k) {
                            std::vector<int> idxA = batchIdx;
                            idxA.push_back(i);
                            idxA.push_back(k);
                            
                            std::vector<int> idxB = batchIdx;
                            idxB.push_back(k);
                            idxB.push_back(j);
                            
                            sum += a.at(idxA) * b.at(idxB);
                        }
                        
                        std::vector<int> idxOut = batchIdx;
                        idxOut.push_back(i);
                        idxOut.push_back(j);
                        result.set(idxOut, sum);
                    }
                }
            } else {
                for (int i = 0; i < shapeA[dim]; ++i) {
                    batchIdx[dim] = i;
                    processBatch(batchIdx, dim + 1);
                }
            }
        };

        std::vector<int> batchIdx(shapeA.size() - 2);
        processBatch(batchIdx, 0);
        
        return result;
    }

    throw std::invalid_argument("Dot only support 1D, 2D, or batched >2D!");
}

// Activation Neuron ReLU //
Tensor Operation::relu(const Tensor& t) {
    auto shape = t.getShape();
    Tensor result(shape);
    
    for (int i = 0; i < t.size(); ++i) {
        std::vector<int> indices(shape.size());
        int remaining = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        double val = t.at(indices);
        result.set(indices, std::max(0.0, val));
    }
    
    return result;
}

// Activation neuron Sigmoid //
Tensor Operation::sigmoid(const Tensor& t) {
    auto shape = t.getShape();
    Tensor result(shape);
    
    for (int i = 0; i < t.size(); ++i) {
        std::vector<int> indices(shape.size());
        int remaining = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        
        double val = t.at(indices);
        result.set(indices, 1.0 / (1.0 + std::exp(-val)));
    }
    
    return result;
}

// Activation neuron Softmax //
Tensor Operation::softmax(const Tensor& t) {
    auto shape = t.getShape();
    
    // For tensor multidimensional, try softmax in end dimensional //
    if (shape.empty()) {
        throw std::invalid_argument("Softmax can't try in the 0 Tensor");
    }
    
    Tensor result(shape);
    
    if (shape.size() == 1) {
        // Simple 1D case //
        double max_val = t.at({0});
        for (int i = 1; i < shape[0]; ++i) {
            max_val = std::max(max_val, t.at({i}));
        }
        
        std::vector<double> exp_vals(shape[0]);
        double sum_exp = 0.0;
        
        for (int i = 0; i < shape[0]; ++i) {
            exp_vals[i] = std::exp(t.at({i}) - max_val);
            sum_exp += exp_vals[i];
        }
        
        for (int i = 0; i < shape[0]; ++i) {
            result.set({i}, exp_vals[i] / sum_exp);
        }
    } else {
        // Multidimensional case - softmax //
        int last_dim = shape[shape.size() - 1];
        int outer_size = t.size() / last_dim;
        
        for (int outer = 0; outer < outer_size; ++outer) {
            // Convert outer index ke multi-dimensional indices //
            std::vector<int> base_indices(shape.size() - 1);
            int remaining = outer;
            for (int j = shape.size() - 2; j >= 0; --j) {
                base_indices[j] = remaining % shape[j];
                remaining /= shape[j];
            }
            
            // Find max for stability numerical //
            std::vector<int> idx = base_indices;
            idx.push_back(0);
            double max_val = t.at(idx);
            
            for (int i = 1; i < last_dim; ++i) {
                idx[idx.size() - 1] = i;
                max_val = std::max(max_val, t.at(idx));
            }
            
            // Compute exp values and sum //
            std::vector<double> exp_vals(last_dim);
            double sum_exp = 0.0;
            
            for (int i = 0; i < last_dim; ++i) {
                idx[idx.size() - 1] = i;
                exp_vals[i] = std::exp(t.at(idx) - max_val);
                sum_exp += exp_vals[i];
            }
            
            // Set normalized values //
            for (int i = 0; i < last_dim; ++i) {
                idx[idx.size() - 1] = i;
                result.set(idx, exp_vals[i] / sum_exp);
            }
        }
    }
    
    return result;
}
