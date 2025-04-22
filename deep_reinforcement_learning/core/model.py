import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Layer(ABC):
    """Abstract base class for neural network layers."""
    
    @abstractmethod
    def build(self, prev_size, next_size):
        """Build the PyTorch layer.
        
        Args:
            prev_size (int): Size of the previous layer (input size).
            next_size (int): Size of the next layer (output size or None for output layer).
        
        Returns:
            nn.Module or list: PyTorch layer module(s). InputLayer and OutputLayer return a single nn.Module,
                              HiddenLayer returns a list of [nn.Linear, activation].
        """
        pass

class InputLayer(Layer):
    """Input layer for the neural network (fully connected)."""
    
    def build(self, prev_size, next_size):
        """Build a linear layer from state_size to next_size.
        
        Args:
            prev_size (int): State size (input dimension).
            next_size (int): Size of the next layer.
        
        Returns:
            nn.Linear: Linear layer.
        """
        return nn.Linear(prev_size, next_size)

class HiddenLayer(Layer):
    """Hidden layer with specified neurons and activation function."""
    
    def __init__(self, num_neurons, activation):
        """Initialize the hidden layer.
        
        Args:
            num_neurons (int): Number of neurons in the layer.
            activation: Activation function (subclass of ActivationFunction).
        
        Raises:
            ValueError: If num_neurons is not positive.
        """
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        self.num_neurons = num_neurons
        self.activation = activation
    
    def build(self, prev_size, next_size):
        """Build a linear layer followed by an activation function.
        
        Args:
            prev_size (int): Size of the previous layer.
            next_size (int): Size of the next layer (ignored, uses num_neurons).
        
        Returns:
            list: [nn.Linear, activation] modules.
        """
        return [nn.Linear(prev_size, self.num_neurons), self.activation.build()]

class OutputLayer(Layer):
    """Output layer for the neural network (fully connected to action_size)."""
    
    def build(self, prev_size, next_size):
        """Build a linear layer to action_size.
        
        Args:
            prev_size (int): Size of the previous layer.
            next_size (int): Action size (output dimension).
        
        Returns:
            nn.Linear: Linear layer.
        """
        return nn.Linear(prev_size, next_size)

class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def build(self):
        """Build the PyTorch activation module.
        
        Returns:
            nn.Module: PyTorch activation module.
        """
        pass

class ReLU(ActivationFunction):
    """ReLU activation function."""
    
    def build(self):
        """Build a ReLU activation module."""
        return nn.ReLU()

class Model(nn.Module):
    """Neural network model built from a list of Layer objects."""
    
    def __init__(self, state_size, action_size, layers=None):
        """Initialize the model.
        
        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            layers (list, optional): List of Layer objects. If None, uses a default architecture
                                   with two hidden layers (256 and 128 neurons).
        
        Raises:
            ValueError: If state_size or action_size is not positive, layers is empty,
                       first layer is not InputLayer, or last layer is not OutputLayer.
        """
        super(Model, self).__init__()
        
        # Validate inputs
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive")
        
        if layers is None:
            # Default architecture: two hidden layers with 256 and 128 neurons
            layers = [
                InputLayer(),
                HiddenLayer(256, ReLU()),
                HiddenLayer(128, ReLU()),
                OutputLayer()
            ]
        
        # Validate layers
        if not layers:
            raise ValueError("Layers list cannot be empty")
        if not isinstance(layers[0], InputLayer):
            raise ValueError("First layer must be InputLayer")
        if not isinstance(layers[-1], OutputLayer):
            raise ValueError("Last layer must be OutputLayer")
        
        self.layers = layers
        self.state_size = state_size
        self.action_size = action_size
        
        # Build the network
        modules = []
        prev_size = state_size
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                # Non-output layer: next_size is the size of the next layer
                next_layer = layers[i + 1]
                if isinstance(next_layer, HiddenLayer):
                    next_size = next_layer.num_neurons
                elif isinstance(next_layer, OutputLayer):
                    next_size = action_size
                else:
                    raise ValueError(f"Unsupported layer type for next layer: {type(next_layer).__name__}")
            else:
                # Output layer: next_size is action_size
                next_size = action_size
            
            built = layer.build(prev_size, next_size)
            if isinstance(built, list):
                # HiddenLayer returns [linear, activation]
                modules.extend(built)
                prev_size = layer.num_neurons
            else:
                # InputLayer or OutputLayer returns a single module
                modules.append(built)
                prev_size = next_size
        
        self.network = nn.Sequential(*modules)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor, expected shape [batch_size, state_size]
                             or [state_size] (single sample).
        
        Returns:
            torch.Tensor: The network output for the specific input state.
        
        Raises:
            ValueError: If input shape is incompatible with state_size.
        """
        # Normalize input shape
        if x.dim() == 1:
            x = x.view(1, -1)  # Convert [state_size] to [1, state_size]
        elif x.dim() != 2 or x.size(1) != self.state_size:
            raise ValueError(f"Expected input shape [batch_size, {self.state_size}], got {x.shape}")
        
        return self.network(x)