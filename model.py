import torch
import torch.nn as nn
import torch.nn.functional as F
class MNIST_CNN(nn.Module):
    """
    Standard CNN model for MNIST digit classification.

    Architecture:
    - Conv2D (16 filters, kernel_size=5, stride=1, padding=2) + ReLU + MaxPooling (kernel_size=2)
    - Conv2D (32 filters, kernel_size=5, stride=1, padding=2) + ReLU + MaxPooling (kernel_size=2)
    - Dropout (0.1)
    - Fully Connected (32 * 7 * 7 -> 96) + ReLU
    - Dropout (0.2)
    - Fully Connected (96 -> 10) + Log Softmax

    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop1 = nn.Dropout2d(p=0.2)  # Dropout to prevent overfitting，Delete the entire channel in the spatial dimension。
        self.fc1 = nn.Linear(32 * 7 * 7, 96)  # Fully connected layer
        self.drop2 = nn.Dropout2d(p=0.1)
        self.fc2 = nn.Linear(96, 10)  # Fully connected layer (maps 96 hidden units to 10 output classes)

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Log softmax output of shape (batch_size, 10).
        """
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # Apply log softmax for classification
        return output

class MNIST_CNN_optuna(nn.Module):
    """
    CNN model for MNIST classification with hyperparameter tuning using Optuna.

    This model searches for optimal:
    - Dropout rates (`dropout_rate1`, `dropout_rate2`)
    - Fully connected layer size (`fc2_input_dim`)

    Architecture:
    - Conv2D (16 filters, kernel_size=5, stride=1, padding=2) + ReLU + MaxPooling (kernel_size=2)
    - Conv2D (32 filters, kernel_size=5, stride=1, padding=2) + ReLU + MaxPooling (kernel_size=2)
    - Dropout (`dropout_rate1`)
    - Fully Connected (32 * 7 * 7 -> `fc2_input_dim`) + ReLU
    - Dropout (`dropout_rate2`)
    - Fully Connected (`fc2_input_dim` -> 10) + Log Softmax
    """
    def __init__(self, trial):
        super(MNIST_CNN_optuna, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)  # First convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5, step=0.1)  # Tune dropout rate for the first dropout layer
        self.drop1=nn.Dropout2d(p=dropout_rate)
        fc2_input_dim = trial.suggest_int("fc2_input_dim", 32, 128, step=32)  # Tune the fully connected layer size
        self.fc1 = nn.Linear(32 * 7 * 7, fc2_input_dim)
        dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.3,step=0.1)  # Tune dropout rate for the second dropout layer
        self.drop2=nn.Dropout2d(p=dropout_rate2)
        self.fc2 = nn.Linear(fc2_input_dim, 10)
    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Log softmax output of shape (batch_size, 10).
        """
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output