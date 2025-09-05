import torch
import torch.nn.functional as F
import numpy as np

class LRP:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.activations = {}
        
    def register_forward_hooks(self):
        """Register hooks to capture forward activations"""
        hooks = []
        
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().clone()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear, torch.nn.ReLU)):
                hooks.append(module.register_forward_hook(save_activation(name)))
        
        return hooks
    
    def lrp_linear_epsilon_rule(self, layer, activation_in, relevance_out, epsilon=1e-6):
        """LRP ε-rule for linear/conv layers"""
        weight = layer.weight.detach()
        
        # Forward pass through layer
        if isinstance(layer, torch.nn.Linear):
            z = torch.mm(activation_in, weight.t())
        elif isinstance(layer, torch.nn.Conv3d):
            z = F.conv3d(activation_in, weight, layer.bias, 
                        layer.stride, layer.padding, layer.dilation, layer.groups)
        
        # Add epsilon for numerical stability
        z += epsilon * (z >= 0).float() - epsilon * (z < 0).float()
        
        # LRP rule: R_i = a_i * sum_j(w_ij * R_j / z_j)
        s = relevance_out / z
        
        if isinstance(layer, torch.nn.Linear):
            c = torch.mm(s, weight)
            relevance_in = activation_in * c
        elif isinstance(layer, torch.nn.Conv3d):
            c = F.conv_transpose3d(s, weight, None, 
                                 layer.stride, layer.padding, 
                                 layer.output_padding, layer.groups, layer.dilation)
            relevance_in = activation_in * c
        
        return relevance_in
    
    def lrp_relu_rule(self, activation_in, relevance_out):
        """LRP rule for ReLU: just pass relevance through"""
        return relevance_out
    
    def compute_lrp_for_embedding_dim(self, input_tensor, embedding_dim):
        """
        Proper LRP implementation
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # Register hooks and do forward pass
        hooks = self.register_forward_hooks()
        
        try:
            with torch.no_grad():
                embedding = self.model(input_tensor)
            
            # Initialize relevance at output layer
            relevance = torch.zeros_like(embedding)
            relevance[0, embedding_dim] = embedding[0, embedding_dim]
            
            # Get layer names in reverse order
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear, torch.nn.ReLU)):
                    layer_names.append(name)
            
            # Propagate relevance backwards through each layer
            for layer_name in reversed(layer_names):
                layer = dict(self.model.named_modules())[layer_name]
                
                if isinstance(layer, torch.nn.ReLU):
                    # ReLU: pass relevance through
                    continue
                    
                elif isinstance(layer, (torch.nn.Conv3d, torch.nn.Linear)):
                    # Get input activation for this layer
                    if layer_name in self.activations:
                        activation_in = self.activations[layer_name]
                        
                        # Find the input to this layer (previous activation)
                        prev_layer_names = layer_names[:layer_names.index(layer_name)]
                        if prev_layer_names:
                            prev_activation = self.activations[prev_layer_names[-1]]
                        else:
                            prev_activation = input_tensor
                        
                        # Apply LRP rule
                        relevance = self.lrp_linear_epsilon_rule(
                            layer, prev_activation, relevance
                        )
            
        finally:
            for hook in hooks:
                hook.remove()
            self.activations.clear()
        
        return relevance.abs()