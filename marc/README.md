# MNIST

## Neural Network

### Stats

**Accuracy: 99.19%**

**Epochs: 200**

**Optimizer for Backpropagation: Adam**

| Layer         | Type | Activation           | Nodes  |
| ------------- |:-------------:|:-------------:| -----:|
| 1      | conv  | relu | Filter output 32 Images |
| 2      | max pooling (2x2)  | - | - |
| 3      | conv  | relu | Filter output 64 Images |
| 4      | max pooling (2x2)  | - | - |
| 5      | NonLinear  | softplus | 1024 |
| 6      | NonLinear  | softplus | 512 |
| 7      | NonLinear  | softplus | 256 |
| 8      | NonLinear  | softplus | 200 |
| 9      | NonLinear  | softplus | 100 |
| 10      | Output  | softplus | 10 |

### Using the pretrained Model
1. Extract the model.zip into the current Folder
2. pip install tensorflow / tensorflow-gpu
3. python mnist_nn.py

### Training from scratch
1. pip install tensorflow / tensorflow-gpu
2. python mnist_nn.py (it will train now)
3. python mnist_nn.py (after is finishes)