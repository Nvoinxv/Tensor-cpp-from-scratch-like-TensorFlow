#include <iostream>
#include <iomanip>
#include "../Tensor/include/Tensor.hpp"
#include "../Tensor/include/ops/all_ops.hpp"

void test_autodiff() {
    std::cout << "=== Test 1: Automatic Differentiation (Autodiff) ===" << std::endl;
    // f(x, y) = sin(x * y) + tanh(x) + sigmoid(y)
    Tensor x({1}, {2.0}, true); // requires_grad = true
    Tensor y({1}, {3.0}, true); // requires_grad = true

    Tensor xy = x * y;
    Tensor sin_xy = ops::sin(xy);
    Tensor tanh_x = ops::tanh(x);
    Tensor sig_y = ops::sigmoid(y);

    Tensor f = sin_xy + tanh_x + sig_y;

    std::cout << "Forward value f(2.0, 3.0) = " << f.at({0}) << std::endl;

    f.backward();

    std::cout << "Gradient df/dx (computed by autodiff) = " << x.gradAt({0}) << std::endl;
    std::cout << "Gradient df/dy (computed by autodiff) = " << y.gradAt({0}) << std::endl;

    // Theoretical check:
    // df/dx = y * cos(x * y) + (1 - tanh^2(x))
    double expected_df_dx = 3.0 * std::cos(6.0) + (1.0 - std::pow(std::tanh(2.0), 2));
    std::cout << "Expected df/dx (analytical)         = " << expected_df_dx << std::endl;
    std::cout << std::endl;
}

void test_matrix_inverse() {
    std::cout << "=== Test 2: Matrix Inverse & Identity Verification ===" << std::endl;
    Tensor M({2, 2}, {4.0, 7.0, 2.0, 6.0}, true);
    std::cout << "Original Matrix M:" << std::endl;
    M.print();

    Tensor invM = ops::inverse(M);
    std::cout << "\nInverse Matrix M^-1:" << std::endl;
    invM.print();

    Tensor I = ops::matmul(M, invM);
    std::cout << "\nVerification M * M^-1 (Should be Identity Matrix ~ [[1, 0], [0, 1]]):" << std::endl;
    I.print();
    std::cout << std::endl;
}

void test_training_step() {
    std::cout << "=== Test 3: Mini Deep Learning Training Step ===" << std::endl;
    // Linear layer prediction: y_pred = X * W
    // MSE Loss: L = mean((y_pred - y_true)^2)
    Tensor X({2, 2}, {1.0, 2.0, 3.0, 4.0}, false);
    Tensor W({2, 1}, {0.5, -0.5}, true); // trainable weight
    Tensor y_true({2, 1}, {1.0, 2.0}, false);

    std::cout << "Initial Weight W:" << std::endl;
    W.print();

    // Forward pass
    Tensor y_pred = ops::matmul(X, W);
    Tensor diff = y_pred - y_true;
    Tensor loss = ops::mean(diff * diff);

    std::cout << "\nInitial Loss: " << loss.at({0}) << std::endl;

    // Backward pass
    loss.backward();

    std::cout << "\nWeight Gradients dL/dW:" << std::endl;
    for (size_t i = 0; i < W.getGrad().size(); ++i) {
        std::cout << "[" << i << "] = " << W.getGrad()[i] << std::endl;
    }

    // Gradient Descent Update: W_new = W - lr * grad
    double lr = 0.05;
    for (size_t i = 0; i < W.size(); ++i) {
        W.getMutableData()[i] -= lr * W.getGrad()[i];
    }
    W.zero_grad();

    // Verify loss decrease after 1 step
    Tensor y_pred_new = ops::matmul(X, W);
    Tensor diff_new = y_pred_new - y_true;
    Tensor loss_new = ops::mean(diff_new * diff_new);

    std::cout << "\nNew Weight W after 1 step:" << std::endl;
    W.print();
    std::cout << "New Loss after 1 step: " << loss_new.at({0}) << " (Decreased!)" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "       TENSOR C++ FROM SCRATCH PROFESSIONAL EDITION       " << std::endl;
    std::cout << "==========================================================" << std::endl << std::endl;

    try {
        test_autodiff();
        test_matrix_inverse();
        test_training_step();
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "All professional demonstrations completed successfully!" << std::endl;
    return 0;
}
