#include <iostream>
#include "../Tensor/include/Tensor.hpp"
#include "../Tensor/include/Operation.hpp"

int main() {
    // We test Tensor with 8 Dimensional //
    Tensor A({8,8}); 
    Tensor B({8,8}); 
    
    int valA = 1, valB = 1;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A({i, j}) = valA++;
            B({i, j}) = valB++;
        }
    }
    
    // Check Tensor A and B //
    std::cout << "Tensor A:" << std::endl;
    A.print();
    std::cout << "\nTensor B:" << std::endl;
    B.print();
    std::cout << std::endl;
    
    // Implementation operation.hpp to the main.cpp //
    Tensor dotResult = Operation::dot(A, B);
    std::cout << "Result from dot product A and B is " << std::endl;
    dotResult.print();
    std::cout << std::endl;

    Tensor addResult = Operation::add(A, B);
    std::cout << "Result from addition Tensor A and B is " << std::endl;
    addResult.print();
    std::cout << std::endl;

    Tensor subResult = Operation::sub(A, B);
    std::cout << "Result from subtraction Tensor A and B is " << std::endl;
    subResult.print();
    std::cout << std::endl;

    Tensor sigmoidResult = Operation::sigmoid(B);
    std::cout << "Result from sigmoid Tensor B is " << std::endl;
    sigmoidResult.print();
    std::cout << std::endl;

    Tensor reluResult = Operation::relu(A);
    std::cout << "Result from relu Tensor A is " << std::endl;
    reluResult.print();
    std::cout << std::endl;

    Tensor mulResult = Operation::mul(A, B);
    std::cout << "Result from Multiple Tensor A and B is " << std::endl;
    mulResult.print();
    std::cout << std::endl;

    return 0;
}
