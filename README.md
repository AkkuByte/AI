# AI Learning Repository

A comprehensive learning resource for understanding neural networks and machine learning fundamentals using Python and NumPy.

## 📋 Project Overview

This repository contains hands-on Jupyter notebooks designed to teach the foundational concepts of artificial intelligence, focusing on neural networks, linear algebra operations, and machine learning algorithms. The notebooks progress from basic neuron operations to complex layer and batch computations.

## 📁 Project Structure

```
AI/
├── Neural Network/
│   ├── neuron.ipynb          # Core neural network concepts and implementations
│   └── Multiplication.ipynb  # Linear algebra operations in neural networks
├── Regression/               # (Future: Regression models and techniques)
└── README.md                 # This file
```

## 📚 Notebooks

### Neural Network Folder

#### 1. **neuron.ipynb**
An introduction to coding neurons and layers from scratch using Python.

**Topics covered:**
- Building a single neuron with 3 inputs
- Building a single neuron with 4 inputs
- Creating layers with multiple neurons
- Using loops to efficiently construct layers
- Understanding weights, biases, and neural computations

**Key concepts:**
- Neuron: A single unit that performs a weighted sum of inputs plus a bias
- Layers: Multiple neurons stacked together to process information
- Weights and Bias: Parameters that the neural network learns

#### 2. **Multiplication.ipynb**
Explores different types of matrix multiplications in the context of neural networks using NumPy.

**Topics covered:**
- **Vector × Vector**: Represents a single neuron computation
- **Vector × Matrix**: Represents computation through a layer
- **Matrix × Matrix**: Represents batch processing multiple data points

**Key concepts:**
- Vector × Vector multiplication produces a scalar output (one neuron)
- Vector × Matrix multiplication produces a vector output (multiple neurons in a layer)
- Matrix × Matrix multiplication handles batches of data efficiently

### Regression Folder
*Currently empty* - Reserved for regression models and techniques to be added in future updates.

## 🛠 Prerequisites

- Python 3.6+
- Jupyter Notebook or JupyterLab
- NumPy library

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Install required packages:**
   ```bash
   pip install numpy jupyter
   ```

## ▶️ Running the Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Navigate to the notebook files:**
   - Open `Neural Network/neuron.ipynb` to start with basic neuron concepts
   - Open `Neural Network/Multiplication.ipynb` to learn about matrix operations

3. **Run cells** sequentially to understand how neural networks process data

## 🎯 Learning Path

1. Start with **neuron.ipynb** to understand the fundamentals of neurons and layers
2. Progress to **Multiplication.ipynb** to see how NumPy optimizes these operations
3. Experiment by modifying weights, biases, and inputs to see how they affect outputs
4. Use concepts learned to build more complex neural networks

## 💡 Key Takeaways

- **Neuron Computation**: `output = sum(inputs × weights) + bias`
- **Vectorization**: NumPy allows efficient computation on vectors and matrices
- **Layers**: Multiple neurons in parallel enable learning complex patterns
- **Batching**: Processing multiple samples simultaneously improves computational efficiency

## 🔬 Example Output

Running the neuron computation:
```python
inputs = [1, 2, 3, 4]
weights = [0.2, 0.8, -0.5, 1]
bias = 2
output = np.dot(inputs, weights) + bias
# Output: 4.7
```

## 📖 Further Learning

- Explore activation functions (ReLU, Sigmoid, Tanh)
- Implement backpropagation for training
- Study optimization algorithms (SGD, Adam)
- Build a complete neural network from scratch
- Experiment with regression and classification tasks

## 📝 Notes

- These notebooks are designed for educational purposes
- Modify and experiment with the code to deepen your understanding
- Add more notebooks to the Regression folder as you progress

## 🤝 Contributing

Feel free to add more notebooks, examples, and improvements to enhance the learning experience.

## 📄 License

This repository is open for educational and learning purposes.

---

**Last Updated:** June 12, 2026
