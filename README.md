# Creating deep learning models on MNIST database using Pytorch 

### MNIST database
http://yann.lecun.com/exdb/mnist/

# LogSoftMax Model
## Steps
### Download the dataset
- use **torchvision.dataset** to download the MNIST digits dataset and transform to Tensors.
- generate train and test loaders of batch size 64 and assign it to variables **trainloader** and **testloader** respectively.
- define the method named model_buid() to build a sequential model.
  - parameters: 
       - **input_size** int: number of input features  
       - **hidden_sizes** array of size 3: array of number of nodes in each of the three hidden layers  
       - **output int**: number of nodes at output layer (or number of classes to classify)  
  - returns model: sequential model  
Apply relu activation between each layer and for the **final layer out apply logSoftmax.**  
Logsoftmax log transformation of softmax output which is given by :$$
 \sigma(x_i) =  \log \cfrac{e^{x_i}}{\sum_{k=1}^{N}{e^{x_k}}}$$ where N is the number of classes  
 more information [here](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax) 
  - using the function we defined, ** we initialize the model to have input size 784, three hidden layers to have 256, 128 and 64 nodes and finally an output layer of size 10 nodes.**
### Define criterion (loss function) and optimizer
  - Define criterion to be [negative likelihood loss](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss) since the network output is log transformed probabilities.
  - Define optimizer to be [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) and set learning rate to 0.003 and pass the parameters of **classifier** model 
### Train the classifier
  - flatten the image tensor from (1 x 28 x 28) to (1, 784)
### Evaluate class probabilities
- transform log transformed softmax output log_ps to exponential and assign it to ps (torch.exp())
- return the top probability and its index from  **ps** (ps.topk())
