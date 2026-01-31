"""
OnlineMLP - Pure NumPy MLP for online learning.

No PyTorch dependency, suitable for low-memory environments.
Uses .npz format for saving/loading weights.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class OnlineMLP:
    """
    Simple 2-layer MLP implemented in pure NumPy.
    
    Architecture: input -> hidden (ReLU) -> output (Sigmoid)
    
    Uses .npz format for persistence (no PyTorch needed).
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer size
        output_dim: Output dimension (1 for binary classification)
        lr: Learning rate
        model_path: Path to save/load weights
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
        lr: float = 0.01,
        model_path: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.model_path = model_path
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        
        self.trained = False
        self.train_count = 0
        
        # Try to load existing weights
        if model_path:
            self.load_model()
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(float)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input array of shape (batch, input_dim) or (input_dim,)
        
        Returns:
            Output probabilities of shape (batch, output_dim)
        """
        X = np.atleast_2d(X)
        
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        
        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        
        return self.a2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.forward(X)
    
    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Single training step with backpropagation.
        
        Args:
            X: Input features (batch, input_dim)
            y: Labels (batch, 1) or (batch,)
        
        Returns:
            Loss value
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y).reshape(-1, 1)
        
        batch_size = X.shape[0]
        
        # Forward pass
        output = self.forward(X)
        
        # Binary cross-entropy loss
        eps = 1e-7
        loss = -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        
        # Backward pass
        # Output layer gradient
        dz2 = output - y  # (batch, 1)
        dW2 = self.a1.T @ dz2 / batch_size
        db2 = np.mean(dz2, axis=0)
        
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(self.z1)
        dW1 = X.T @ dz1 / batch_size
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        self.train_count += batch_size
        self.trained = True
        
        return float(loss)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> float:
        """
        Train the model.
        
        Args:
            X: Training features (n_samples, input_dim)
            y: Training labels (n_samples,)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print progress
        
        Returns:
            Final loss
        """
        X = np.atleast_2d(X)
        y = np.array(y).flatten()
        
        n_samples = X.shape[0]
        loss = 0.0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        # Save after training
        if self.model_path:
            self.save_model()
        
        return loss
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save model weights to .npz file."""
        save_path = path or self.model_path
        if save_path is None:
            return
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            str(save_path_obj),
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            train_count=self.train_count,
        )
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load model weights from .npz file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path or self.model_path
        if load_path is None:
            return False
        
        load_path = Path(load_path)
        if not load_path.exists():
            return False
        
        try:
            data = np.load(load_path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.input_dim = int(data['input_dim'])
            self.hidden_dim = int(data['hidden_dim'])
            self.output_dim = int(data['output_dim'])
            self.train_count = int(data.get('train_count', 0))
            self.trained = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def accuracy(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate accuracy."""
        preds = self.predict_class(X, threshold)
        y = np.array(y).flatten()
        return float(np.mean(preds.flatten() == y))
