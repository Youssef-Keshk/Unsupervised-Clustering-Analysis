import numpy as np
from typing import Optional, Tuple, List

class Autoencoder:
    """
    Fully connected Autoencoder implementation from scratch.
    """
    
    def __init__(
            self,
            input_dim: int,
            encoding_dims: List[int],
            bottleneck_dim: int,
            activation: str = 'relu',
            learning_rate: float = 0.01,
            batch_size: int = 32,
            epochs: int = 100,
            l2_lambda: float = 0.01,
            lr_decay: float = 0.95,
            random_state: Optional[int] = None
        ):

        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.lr_decay = lr_decay
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Build architecture
        self.encoder_dims = [input_dim] + encoding_dims + [bottleneck_dim]
        self.decoder_dims = [bottleneck_dim] + encoding_dims[::-1] + [input_dim]
        
        # Initialize weights and biases
        self._initialize_weights()
        
        # Training history
        self.train_loss_history = []
        
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        self.encoder_weights = []
        self.encoder_biases = []
        
        for i in range(len(self.encoder_dims) - 1):
            # He initialization
            w = np.random.randn(self.encoder_dims[i], self.encoder_dims[i+1]) * np.sqrt(2.0 / self.encoder_dims[i])
            b = np.zeros((1, self.encoder_dims[i+1]))
            self.encoder_weights.append(w)
            self.encoder_biases.append(b)
        
        self.decoder_weights = []
        self.decoder_biases = []
        
        for i in range(len(self.decoder_dims) - 1):
            w = np.random.randn(self.decoder_dims[i], self.decoder_dims[i+1]) * np.sqrt(2.0 / self.decoder_dims[i])
            b = np.zeros((1, self.decoder_dims[i+1]))
            self.decoder_weights.append(w)
            self.decoder_biases.append(b)

    
    def _activation_forward(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_backward(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activation_forward(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, List, List]:
        """
        Forward pass through the network.
        """
        activations = [X]
        z_values = []
        
        # Encoder forward pass
        current = X
        for i in range(len(self.encoder_weights)):
            z = current @ self.encoder_weights[i] + self.encoder_biases[i]
            z_values.append(z)
            current = self._activation_forward(z)
            activations.append(current)
        
        # Decoder forward pass
        for i in range(len(self.decoder_weights) - 1):
            z = current @ self.decoder_weights[i] + self.decoder_biases[i]
            z_values.append(z)
            current = self._activation_forward(z)
            activations.append(current)
        
        # Output layer (no activation)
        z = current @ self.decoder_weights[-1] + self.decoder_biases[-1]
        z_values.append(z)
        output = z  # Linear activation for output
        activations.append(output)
        
        return output, activations, z_values
    
    def _backward_pass(
        self,
        X: np.ndarray,
        output: np.ndarray,
        activations: List,
        z_values: List
    ) -> Tuple[List, List, List, List]:
        """
        Backward pass through the network.
        """
        n_samples = X.shape[0]
        
        # Output layer gradient (MSE loss derivative)
        delta = (output - X) / n_samples
        
        # Decoder gradients
        decoder_weight_grads = []
        decoder_bias_grads = []
        
        # Start from last decoder layer
        layer_idx = len(activations) - 2
        
        for i in range(len(self.decoder_weights) - 1, -1, -1):
            # Gradient for weights and biases
            dW = activations[layer_idx].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Add L2 regularization
            dW += self.l2_lambda * self.decoder_weights[i]
            
            decoder_weight_grads.insert(0, dW)
            decoder_bias_grads.insert(0, db)
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.decoder_weights[i].T) * self._activation_backward(z_values[layer_idx - 1])
            else:
                delta = delta @ self.decoder_weights[i].T
            
            layer_idx -= 1
        
        # Encoder gradients
        encoder_weight_grads = []
        encoder_bias_grads = []
        
        for i in range(len(self.encoder_weights) - 1, -1, -1):
            delta = delta * self._activation_backward(z_values[layer_idx])
            
            # Gradient for weights and biases
            dW = activations[layer_idx].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Add L2 regularization
            dW += self.l2_lambda * self.encoder_weights[i]
            
            encoder_weight_grads.insert(0, dW)
            encoder_bias_grads.insert(0, db)
            
            # Propagate error to previous layer
            if i > 0:
                delta = delta @ self.encoder_weights[i].T
            
            layer_idx -= 1
        
        return encoder_weight_grads, encoder_bias_grads, decoder_weight_grads, decoder_bias_grads
    
    def fit(self, X: np.ndarray, verbose: bool = True) -> 'Autoencoder':
        """
        Train the autoencoder.
        """
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                
                # Forward pass
                output, activations, z_values = self._forward_pass(batch_X)
                
                # Compute loss (MSE + L2 regularization)
                mse_loss = np.mean((output - batch_X) ** 2)
                l2_loss = 0
                for w in self.encoder_weights + self.decoder_weights:
                    l2_loss += np.sum(w ** 2)
                l2_loss *= self.l2_lambda / 2
                
                batch_loss = mse_loss + l2_loss
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                enc_w_grads, enc_b_grads, dec_w_grads, dec_b_grads = self._backward_pass(
                    batch_X, output, activations, z_values
                )
                
                # Update weights
                for j in range(len(self.encoder_weights)):
                    self.encoder_weights[j] -= self.learning_rate * enc_w_grads[j]
                    self.encoder_biases[j] -= self.learning_rate * enc_b_grads[j]
                
                for j in range(len(self.decoder_weights)):
                    self.decoder_weights[j] -= self.learning_rate * dec_w_grads[j]
                    self.decoder_biases[j] -= self.learning_rate * dec_b_grads[j]
            
            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            self.train_loss_history.append(avg_loss)
            
            # Learning rate decay
            self.learning_rate = self.initial_lr * (self.lr_decay ** epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, LR: {self.learning_rate:.6f}")
        
        return self
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data to bottleneck representation.
        """
        current = X
        for i in range(len(self.encoder_weights)):
            z = current @ self.encoder_weights[i] + self.encoder_biases[i]
            current = self._activation_forward(z)
        return current
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode from bottleneck representation.
        """
        current = encoded
        for i in range(len(self.decoder_weights) - 1):
            z = current @ self.decoder_weights[i] + self.decoder_biases[i]
            current = self._activation_forward(z)
        
        # Output layer (linear)
        output = current @ self.decoder_weights[-1] + self.decoder_biases[-1]
        return output
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data (encode then decode).
        """
        encoded = self.encode(X)
        return self.decode(encoded)
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error (MSE).
        """
        reconstructed = self.reconstruct(X)
        return np.mean((X - reconstructed) ** 2)
