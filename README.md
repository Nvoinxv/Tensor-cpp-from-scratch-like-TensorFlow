## 1.What is Tensor?

A Tensor is basically a way to manage shapes in all dimensions. This is super important, especially in frameworks like TensorFlow and PyTorch for building deep learning algorithms.

You can think of a Tensor as a combination of scalar, vector, and matrix. To really understand Tensors, you need to know some linear algebra. If you try to make deep learning without Tensors, it’s gonna be very hard to handle high-dimensional calculations. That’s why we use Tensors—to manage and calculate across all dimensions easily.

## 2.Example Case

Let’s say you want to build a convolutional algorithm for image classification. An image is usually 3D data (width, height, channels). If your algorithm only supports a 2D vector and you try to train with image data, it’s gonna fail because the dimensions don’t match. That’s where Tensors save the day.

## 3.How to Run My Project
```bash
# You need run in powershell like this
# for avoiding launch.json in vscode 
# before you run this terminal, you must in the folder Tensor!
g++ main.cpp src/Operation.cpp src/Tensor.cpp -o main.exe
```

## 4.Closing
So yeah, this is my documentation for building Tensors from scratch with C++ for deep learning. And sorry my english explained is very bad so i need you understand my explain about Tensor from scratch with cpp. Hope you enjoy and learn something from it. Peace out! 
