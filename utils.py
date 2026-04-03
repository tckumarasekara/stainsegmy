import torch


def weights_init(m):
    """
    Custom weight initialization function for PyTorch models. This function initializes 
    the weights of convolutional layers and batch normalization layers using a normal distribution. 
    For convolutional layers, the weights are initialized with a mean of 0.0 and a standard deviation of 0.02. 
    For batch normalization layers, the weights are initialized with a mean of 1.0 and a standard deviation of 0.02, 
    while the biases are set to zero.

    parameters:
    - m: A PyTorch module (layer) to be initialized.

    returns:
    - None
    """
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, torch.nn.Sequential):
        for val in m:
            weights_init(val)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


