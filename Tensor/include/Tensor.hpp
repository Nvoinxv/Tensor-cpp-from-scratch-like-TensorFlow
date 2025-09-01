#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <functional>
#include <utility>

class Tensor {
private:
    std::vector<double> data;      
    std::vector<int> shape;         
    std::vector<int> strides;      
    int total_size;                 

    // Private helper methods //
    int computeTotalSize(const std::vector<int>& shape) const;
    int flattenIndex(const std::vector<int>& indices) const;
    void computeStrides();          
    void printRecursive(const std::vector<int>& indices, int dim) const;
    
public:
    // Constructors //
    Tensor(const std::vector<int>& shape);   
    Tensor(const std::vector<int>& shape, const std::vector<double>& values);

    // Operator overloads untuk akses elemen //
    double& operator()(const std::initializer_list<int>& indices);
    const double& operator()(const std::initializer_list<int>& indices) const;

    // Getters //
    std::vector<int> getShape() const;
    int size() const;
    int rank() const;                       
    bool isScalar() const;     // Check if tensor is scalar //
    bool isEmpty() const;      // Check if tensor is empty //
    
    // Element access methods
    double at(const std::vector<int>& indices) const;      // read-only //
    double& at(const std::vector<int>& indices);           // read-write //
    void set(const std::vector<int>& indices, double value);
    void apply(const std::function<double(double)>& func);

    // Direct data access //
    const std::vector<double>& getData() const { return data; }
    std::vector<double>& getMutableData() { return data; }
    const std::vector<int>& getStrides() const { return strides; }  
    
    // operations //
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor slice(const std::vector<std::pair<int, int>>& ranges) const;

    // Utility methods //
    void print() const;
    
    // Memory optimization methods //
    void reserve(size_t capacity) { data.reserve(capacity); }
    void shrink_to_fit() { data.shrink_to_fit(); }
};

#endif