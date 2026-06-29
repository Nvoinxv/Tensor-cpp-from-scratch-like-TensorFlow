#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <initializer_list>
#include <iostream>

class Tensor;

struct TensorImpl {
    std::vector<double> data;      
    std::vector<double> grad;
    std::vector<int> shape;         
    std::vector<int> strides;      
    int total_size;                 
    bool requires_grad;

    // Autodiff computation graph
    std::vector<Tensor> parents;
    std::function<void()> backward_fn;

    TensorImpl(const std::vector<int>& shape, bool req_grad = false);
    TensorImpl(const std::vector<int>& shape, const std::vector<double>& values, bool req_grad = false);

    void computeStrides();          
    int computeTotalSize(const std::vector<int>& shape) const;
    int flattenIndex(const std::vector<int>& indices) const;
};

class Tensor {
private:
    std::shared_ptr<TensorImpl> impl;

    void printRecursive(const std::vector<int>& indices, int dim) const;

public:
    // Constructors //
    Tensor();
    Tensor(const std::vector<int>& shape, bool requires_grad = false);   
    Tensor(const std::vector<int>& shape, const std::vector<double>& values, bool requires_grad = false);
    explicit Tensor(std::shared_ptr<TensorImpl> ptr);

    // Static Factory Methods //
    static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<int>& shape, bool requires_grad = false);
    static Tensor randn(const std::vector<int>& shape, double mean = 0.0, double stddev = 1.0, bool requires_grad = false);

    // Operator overloads untuk akses elemen //
    double& operator()(const std::initializer_list<int>& indices);
    const double& operator()(const std::initializer_list<int>& indices) const;

    // Getters //
    std::vector<int> getShape() const;
    int size() const;
    int rank() const;                       
    bool isScalar() const;     
    bool isEmpty() const;      
    std::shared_ptr<TensorImpl> getImpl() const { return impl; }

    // Element access methods
    double at(const std::vector<int>& indices) const;      
    double& at(const std::vector<int>& indices);           
    void set(const std::vector<int>& indices, double value);
    void apply(const std::function<double(double)>& func);

    // Direct data access //
    const std::vector<double>& getData() const;
    std::vector<double>& getMutableData() const;
    const std::vector<int>& getStrides() const;  
    
    // Autodiff / Gradient methods //
    bool requiresGrad() const;
    void setRequiresGrad(bool req);
    const std::vector<double>& getGrad() const;
    std::vector<double>& getMutableGrad() const;
    double& gradAt(const std::vector<int>& indices) const;
    void zero_grad();
    void backward();

    // Operations //
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor slice(const std::vector<std::pair<int, int>>& ranges) const;

    // Operator Overloads untuk kemudahan sintaks //
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor operator+(double val) const;
    Tensor operator-(double val) const;
    Tensor operator*(double val) const;
    Tensor operator/(double val) const;

    // Utility methods //
    void print() const;
    
    // Memory optimization methods //
    void reserve(size_t capacity);
    void shrink_to_fit();
};

#endif