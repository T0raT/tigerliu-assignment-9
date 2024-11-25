import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.colors import LinearSegmentedColormap

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros((1, output_dim))
        
        # Storage for visualization
        self.hidden_features = None
        self.gradients = {'W1': None, 'b1': None, 'W2': None, 'b2': None}

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return x

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return np.ones_like(x)

    def forward(self, X):
        # Store input
        self.X = X
        
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.activation(self.z1)
        self.hidden_features = self.h1
        
        # Output layer
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.out = self.z2  # Linear activation for output
        
        return self.out

    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer gradients
        delta2 = self.out - y
        
        # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1)
        
        # Store gradients
        self.gradients['W2'] = np.dot(self.h1.T, delta2) / m
        self.gradients['b2'] = np.sum(delta2, axis=0, keepdims=True) / m
        self.gradients['W1'] = np.dot(X.T, delta1) / m
        self.gradients['b1'] = np.sum(delta1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.lr * self.gradients['W2']
        self.b2 -= self.lr * self.gradients['b2']
        self.W1 -= self.lr * self.gradients['W1']
        self.b1 -= self.lr * self.gradients['b1']

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_hidden, ax_input, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Set titles
    ax_hidden.set_title(f'Hidden Space at Step {frame*10}')
    ax_input.set_title(f'Input Space at Step {frame*10}')
    ax_gradient.set_title(f'Gradients at Step {frame*10}')
    
    # Training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Plot
    hidden_features = mlp.hidden_features
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                     c=y.ravel(), cmap='bwr', alpha=0.7)
    
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)
    
    # Add hyperplane
    xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2) / mlp.W2[2]
    surf = ax_hidden.plot_surface(xx, yy, zz, alpha=0.2, cmap='autumn')
    
    ax_hidden.view_init(elev=20, azim=-70)
    ax_hidden.set_box_aspect([1,1,1])

    # Input Space Plot
    x_min, x_max = -3, 3
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))
    
    # Get network predictions
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_binary = np.where(Z > 0, 1, -1)

    # Define vibrant red and blue colors
    colors = [(0.0, (0.0, 0.2, 1.0)),    # Vibrant blue
             (1.0, (1.0, 0.0, 0.0))]     # Vibrant red
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    
    # Plot decision boundary with vibrant colors and adjusted alpha
    ax_input.pcolormesh(xx, yy, Z_binary, cmap=custom_cmap, alpha=0.5)
    
    # Plot scatter points
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='black', linewidth=0.5)
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)


    # Gradient Visualization
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    
    # Node positions
    nodes = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y':  (1.0, 0.0)
    }
    
    # Draw nodes
    for name, pos in nodes.items():
        circle = plt.Circle(pos, 0.05, color='blue')
        ax_gradient.add_patch(circle)
        ax_gradient.text(pos[0]-0.02, pos[1]+0.07, name, fontsize=10)
    
    # Draw edges with gradient-based thickness
    max_gradient = max(np.abs(mlp.gradients['W1']).max(), 
                      np.abs(mlp.gradients['W2']).max())
    
    # Input to hidden connections
    for i, input_name in enumerate(['x1', 'x2']):
        for j, hidden_name in enumerate(['h1', 'h2', 'h3']):
            gradient = np.abs(mlp.gradients['W1'][i, j])
            width = 0.5 + 2 * gradient / max_gradient
            ax_gradient.plot([nodes[input_name][0], nodes[hidden_name][0]],
                           [nodes[input_name][1], nodes[hidden_name][1]],
                           color='purple', alpha=0.6, linewidth=width)
    
    # Hidden to output connections
    for j, hidden_name in enumerate(['h1', 'h2', 'h3']):
        gradient = np.abs(mlp.gradients['W2'][j, 0])
        width = 0.5 + 2 * gradient / max_gradient
        ax_gradient.plot([nodes[hidden_name][0], nodes['y'][0]],
                        [nodes[hidden_name][1], nodes['y'][1]],
                        color='purple', alpha=0.6, linewidth=width)
    
    ax_gradient.set_aspect('equal')
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_hidden=ax_hidden, 
                                   ax_input=ax_input, ax_gradient=ax_gradient,
                                   X=X, y=y), 
                       frames=step_num//10, repeat=False)
    
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)