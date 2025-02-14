# Neural Network from Scratch (aka SigmoidNN)

This project implements a simple yet powerful **neural network** using scary mathematical operations, such as the sigmoid activation function, cross-entropy loss, and matrix operations😖. The network has been tested with the MNIST dataset of handwritten digits and achieved an **97.99% accuracy** with the chosen parameters.

You can customize the layers, activation functions, and data to train and test the network for your specific needs.

---

#### Checkout our python library [sigmoidNN](https://pypi.org/project/sigmoidNN/)
```python
python3 -m pip install sigmoidNN
```

## Features

- **Fully Customizable Neural Network:** Define your own architecture, including the number of layers and neurons.
- **Basic Mathematical Foundations:** Built using fundamental concepts, including the sigmoid activation function and cross-entropy cost function.
- **Pre-trained Models:** Explore trained instances stored in the `src/trained_instances` folder.
- **High Accuracy:** The network achieves up to **97.99% accuracy** on the MNIST dataset with the default configuration.
- **Modular Code:** Easy-to-understand and modify for experimentation with different datasets and configurations.

---

## Folder Structure

```
project-root/
|
├── src/                # Source code for the neural network
│   ├── network.py      # Core neural network implementation
│   ├── utils.py        # Helper functions for data processing
│   ├── train.py        # Training script
│   └── test.py         # Testing script
|
├── data/               # Folder for datasets (e.g., MNIST)
│   └── mnist/          # MNIST dataset files
|
├── src/trained_instances/   # Pre-trained model instances
│   └── net[<accuracy>].json # Example trained model for MNIST
|
└── README.md           # Project documentation (this file)
```

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- NumPy
- dataset (you can find one in `/data` folder)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Anas-github-acc/SigmoidNN
   cd SigmoidNN
   ```
2. Create the environment (it's recommended to use a virtual environment)::
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For MacOS/Linux
   venv\Scripts\activate      # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Qucik Start

### [Run on the Colab](https://colab.research.google.com/drive/1K5M0E2IBNb1WNcraN5uXMP7E21xP1_by?usp=sharing)
- download the [mnist.data.gz](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1l_zwAKQTlZPib4xhVGriDpVqjaXkXZk-%2Fview%3Fusp%3Ddrive_link) and upload in the colab file
- or upload your own data, edit the path in cell 3 and adjust the parameters like layers
- run all the colab

### Run the network with the MNIST dataset:
```bash
python src/main.py
```
This will save the trained model in the `src/trained_instances/` folder ( their you will find pre-trained result's instance with 96-98% accuracy with chosen parameters
You can load these instance and use

#### Load the instance
Loading the trained instance in `src/trained_instances/`:
```bash
from network import load
load(/Path /to /Instance)
```

#### Customizing the Network
To customize the network, modify the parameters in the `network.py` file, such as:

- Number of layers & Neurons per layer (this take input in list)
- Learning rate
- Regulation
- Batch size
- Activation function

---

## Results

The network achieved the following results on the MNIST dataset:

- **Accuracy:** 97.99%
- **Training Parameters:**
  - Activation function: CrossEntropyCost
  - Regularization: 0.7
  - Learning Rate: 0.5
  - Batch Size: 10
  - Number of Epochs: 15
  - Architecture: [784, 100, 10] (Input Layer -> 1 Hidden Layers -> Output Layer)

Pre-trained results are available in `src/trained_instances/net[<accuracy>].json`. You can load and test this model for verification or use.

---

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## Acknowledgments

I highly recommend the book by Michael Nielsen introducing neural networks and deep learning: https://goo.gl/Zmczdy
