# Deep Neural Network – Cat vs Non-Cat Classifier

## Overview
This project implements a **deep feedforward neural network from scratch using NumPy** to classify images as **cat (1)** or **non-cat (0)**. It is based on the classic DeepLearning.AI (Andrew Ng) Week 4 assignment and is designed to clearly demonstrate how a deep neural network works internally—**forward propagation, backpropagation, cost computation, and parameter updates**—without using high-level ML libraries like TensorFlow or PyTorch.

---

## What the Model Does
- Takes a **64×64 RGB image** as input
- Flattens it into a **12288-dimensional vector**
- Passes it through multiple hidden layers using **ReLU activation**
- Outputs a **single probability** using **Sigmoid activation**
- Classifies the image as:
  - `1` → Cat 🐱
  - `0` → Not Cat ❌

---

## Dataset
The dataset is stored in **HDF5 (.h5) files**:
- `train_catvnoncat.h5`
- `test_catvnoncat.h5`

Each file contains:
- Image data (`train_set_x`, `test_set_x`)
- Labels (`train_set_y`, `test_set_y`)
- Class names (`list_classes`)

Images are normalized by dividing pixel values by **255**.

---

## Network Architecture
```python
layers_dims = [12288, 20, 7, 5, 1]
```

### Meaning of each layer
| Layer | Neurons | Purpose |
|-----|--------|--------|
| Input | 12288 | Flattened 64×64×3 image |
| Hidden 1 | 20 | Learns low-level features |
| Hidden 2 | 7 | Learns feature combinations |
| Hidden 3 | 5 | Learns high-level patterns |
| Output | 1 | Outputs probability of cat |

---

## Activations Used
- **ReLU** → All hidden layers
- **Sigmoid** → Output layer

---

## Initialization
Weights are initialized using **He initialization**, which is essential for ReLU networks:

```python
W[l] ~ N(0, sqrt(2 / n[l-1]))
```

This prevents vanishing gradients and allows faster, stable learning.

Biases are initialized to zero.

---

## Training Process
Training follows these steps for each iteration:
1. Forward propagation through all layers
2. Cost computation using **binary cross-entropy**
3. Backpropagation to compute gradients
4. Gradient descent parameter updates

Key hyperparameters:
- Learning rate: `0.0075`
- Iterations: `2500`

---

## Prediction & Evaluation
After training, the model:
- Predicts labels for training and test sets
- Converts probabilities to binary outputs using a **0.5 threshold**
- Prints accuracy

Typical results (with correct initialization):
- **Training accuracy:** ~99%
- **Test accuracy:** ~70–80%

---

## Why This Project Matters
- Shows how deep learning works **under the hood**
- Builds intuition for:
  - weight initialization
  - activation functions
  - vanishing gradients
  - bias vs variance
- Forms a strong foundation before using frameworks

---

## Requirements
- Python 3.x
- NumPy
- h5py
- Matplotlib (optional, for visualization)

---

## Summary
This project is a **from-scratch implementation of a deep neural network** for binary image classification. It prioritizes **clarity and learning** over abstraction, making it ideal for students who want to truly understand deep learning mechanics.

---

*Built for learning, not shortcuts.*

