# neural-net-from-scratch-numpy

A pure Python/NumPy implementation of a 2-layer Neural Network designed to solve the non-linear "Spiral Dataset" challenge.

## Overview
This project demonstrates the fundamental mathematics of Deep Learning without using high-level frameworks like TensorFlow or PyTorch. It implements:
- **Forward Propagation**: Linear transformations with ReLU activation.
- **Backpropagation**: Manual gradient calculation for weights and biases.
- **Optimization**: Gradient Descent with L2 Regularization.
- **Softmax/Cross-Entropy**: For multi-class classification.

## Spiral Data
The model is tested on a generated spiral dataset with 3 classes. Because the classes wrap around each other, a simple linear classifier would fail. This NN uses a hidden layer of 100 neurons to learn the complex decision boundaries.

![Spiral Data](/fig/Spiral_data.png)

![Decision boundary](/fig/Decision_boundary.png)

## Libraries
- **NumPy**: For all matrix operations and linear algebra.
- **Matplotlib**: For visualizing the decision boundary.

##  How to Run
```bash
# Clone the repo
git clone [your-repo-link]
# Install dependencies
pip install numpy matplotlib
# Run the model
python nn_from_scratch.py
