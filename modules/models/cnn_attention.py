"""
CNN1D model for time series forecasting.
"""

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from ..utils.utils import get_available_device, forecast_support_torch, clear_memory
from ._base import BaseModelWrapper

class CustomModel(nn.Module):
    def __init__(self, window_size, n_features):
        super(CustomModel, self).__init__()
        self.window_size = window_size
        self.n_features = n_features

        self.conv1_features = 128
        self.conv1_kernel_size = 5

        self.conv2_features = 128
        self.conv2_kernel_size = 5

        self.conv3_features = 256
        self.conv3_kernel_size = 5

        self.conv4_features = 256
        self.conv4_kernel_size = 5

        self.hidden1_features = 256
        self.hidden2_features = 128

        self.dropout = 0.5

        self.conv1 = nn.Conv1d(in_channels=self.n_features, out_channels=self.conv1_features, 
                              kernel_size=self.conv1_kernel_size, padding="same")
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=self.conv1_features, out_channels=self.conv2_features,
                              kernel_size=self.conv2_kernel_size, padding="same")
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=self.conv2_features, out_channels=self.conv3_features,
                               kernel_size=self.conv3_kernel_size, padding="same")
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=self.conv3_features, out_channels=self.conv4_features,
                               kernel_size=self.conv4_kernel_size, padding="same")
        self.relu4 = nn.ReLU()

        self.attention = nn.MultiheadAttention(embed_dim=self.window_size, num_heads=self.window_size, batch_first=True)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(self.conv4_features * self.window_size, self.hidden1_features)
        self.relu3 = nn.ReLU()
        self.ln3 = nn.LayerNorm(self.hidden1_features)
        self.dropout = nn.Dropout(self.dropout)

        self.fc2 = nn.Linear(self.hidden1_features, self.hidden2_features)
        self.relu4 = nn.ReLU()
        self.ln4 = nn.LayerNorm(self.hidden2_features)

        self.fc3 = nn.Linear(self.hidden2_features, self.n_features)
        self._init_weights()

    def _init_weights(self):
        # Apply Glorot initialization (Xavier initialization)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Shape: (batch_size, window_size, n_features) -> (batch_size, n_features, window_size)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x, _ = self.attention(x, x, x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.ln3(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu4(x)
        x = self.ln4(x)
        
        x = self.fc3(x)
        return x

class CNNAttention(BaseModelWrapper):
    """
    CNN1D model for time series forecasting.
    """
    name = 'cnn_attention'

    def __init__(self, n_features: int = 1, **kwargs):
        super().__init__(**kwargs)
        # Set the input is generator object
        self.is_generator = True

        # Univariate only have 1 feature
        self.n_features = n_features

        # Training parameters
        self.epochs = kwargs.get('epochs', 100)
        self.patience = kwargs.get('patience', 5)
        self.learning_rate = kwargs.get('lr', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)

        # Loss history
        self.losses = []

        # Get the available device
        self.device = get_available_device()

    def create_model(self, n_features):
        ...

    def fit(self, generator, x, y):
        '''
        generator: WindowGenerator object
            data: (batch_size, window_size, n_features)
            target: (batch_size, n_features)
        '''
        # Initialize the model
        self.model = CustomModel(window_size=generator.window_size, n_features=self.n_features).to(self.device)

        # Define the loss function and optimizer
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = Adam(self.model.parameters(),
                         lr=self.learning_rate, weight_decay=self.weight_decay)

        # Training loop on epochs
        for epoch in range(self.epochs):
            # Training loop on batches
            _datagen = tqdm(generator, desc=f'Epoch {epoch+1}/{self.epochs}')
            for data, target in _datagen:
                # Convert the numpy array to tensor
                data = torch.tensor(data, dtype=torch.float32).to(self.device)
                target = torch.tensor(target, dtype=torch.float32).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Show the loss
                _datagen.set_postfix(loss=loss.item())

            # Save the loss
            self.losses.append(loss.item())

            # Early stopping
            if epoch > self.patience and max(self.losses[-self.patience:]) == self.losses[-1]:
                print('⛔ Early stopping')
                break
        self.model.eval()

    def predict(self, generator, x):
        raise NotImplementedError(
            'Prediction is currently not supported for CNN1D model.')

    def forecast(self, x, steps):
        '''
        x: np.array
            Shape: (window_size, n_features)
        steps: int
        '''
        # Convert the numpy array to tensor
        x_tensor = torch.tensor(
            x.copy(), dtype=torch.float32).unsqueeze(0).to(self.device)

        return forecast_support_torch(self.model, x_tensor, steps)

    def summary(self):
        print(f'{self.name} is not support summary because model is dynamic created.')

    def reset(self):
        del self.model
        clear_memory()

    def get_params(self):
        return {}
