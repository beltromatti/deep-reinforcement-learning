import torch
import logging
from .model import Model, InputLayer, OutputLayer, HiddenLayer, ReLU

def setup_logger(name, level=logging.INFO):
    """Setup the package logger.
    
    Args:
        name (str): Logger name.
        level: Logging level (e.g., logging.INFO).
    
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_model(model: Model, filepath: str = "model_1.model") -> None:
    """Save the model, including architecture and weights.
    
    Args:
        model (Model): A deep neural network.
        filepath (str): Path to save the model.
    
    Raises:
        TypeError: If model is not an instance of Model.
    """
    if not isinstance(model, Model):
        raise TypeError("Model must be an instance of Model class")
    
    # Create a dictionary to store model info
    model_info = {
        'state_size': model.state_size,
        'action_size': model.action_size,
        'layers': [
            {
                'type': layer.__class__.__name__,
                'num_neurons': getattr(layer, 'num_neurons', None),
                'activation': layer.activation.__class__.__name__ if hasattr(layer, 'activation') else None
            } for layer in model.layers
        ],
        'model_state': model.state_dict()  # Save weights separately for flexibility
    }
    
    # Save the entire model info
    torch.save(model_info, filepath)
    logger = setup_logger(__name__)
    logger.info(f"Model saved to: {filepath}")

def load_model(filepath: str) -> Model:
    """Load a PyTorch model with its architecture and weights.
    
    Args:
        filepath (str): Path to the saved model file.
    
    Returns:
        Model: Deep neural network with loaded architecture and weights.
    
    Raises:
        FileNotFoundError: If the filepath does not exist.
        KeyError: If required model info is missing.
        ValueError: If layer or activation types are unsupported.
    """
    # Load the model info
    model_info = torch.load(filepath, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    
    # Extract parameters
    state_size = model_info['state_size']
    action_size = model_info['action_size']
    layer_configs = model_info['layers']
    
    # Reconstruct layers
    layers = []
    for config in layer_configs:
        layer_type = config['type']
        
        if layer_type == 'InputLayer':
            layers.append(InputLayer())
        elif layer_type == 'HiddenLayer':
            num_neurons = config['num_neurons']
            activation_type = config['activation']
            if activation_type == 'ReLU':
                activation = ReLU()
            else:
                raise ValueError(f"Unsupported activation type: {activation_type}")
            layers.append(HiddenLayer(num_neurons, activation))
        elif layer_type == 'OutputLayer':
            layers.append(OutputLayer())
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    # Initialize the model
    model = Model(state_size, action_size, layers)
    
    # Load the weights
    model.load_state_dict(model_info['model_state'])
    
    logger = setup_logger(__name__)
    logger.info(f"Model loaded from: {filepath}")
    return model