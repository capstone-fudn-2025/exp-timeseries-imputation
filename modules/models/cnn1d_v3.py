"""
CNN1D model for time series forecasting.
"""

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from loguru import logger
from ..utils.utils import get_available_device, forecast_support_torch, clear_memory
from ._base import BaseModelWrapper


class CNN1D(nn.Module):
    def __init__(self, n_features: int, window_size: int, config: dict):
        super(CNN1D, self).__init__()

        self.n_features = n_features
        self.window_size = window_size

        self.conv0_features = config.get('conv1_features', 256)
        self.conv0_kernel_size = config.get('conv1_kernel_size', 5)

        self.conv1_features = config.get('conv1_features', 256)
        self.conv1_kernel_size = config.get('conv1_kernel_size', 5)

        self.conv2_features = config.get('conv2_features', 256)
        self.conv2_kernel_size = config.get('conv2_kernel_size', 5)

        self.hidden1_features = config.get('hidden1_features', 256)
        self.hidden2_features = config.get('hidden2_features', 128)

        self.dropout = config.get('dropout', 0.5)

        self.initialize_method = config.get(
            'initialize_method', "glorot")  # glorot or he
        if self.initialize_method not in ['glorot', 'he']:
            logger.warning(
                'Invalid initialize method. Use default Glorot initialization.')
            self.initialize_method = 'glorot'

        # Define the layers
        self.layers = [
            # Shape: (batch_size, n_features, window_size) -> (batch_size, conv1_features, window_size)
            nn.Conv1d(in_channels=n_features, out_channels=self.conv0_features,
                      kernel_size=self.conv0_kernel_size, padding=(self.conv0_kernel_size // 2)),
            nn.ReLU(),

            # Shape: (batch_size, conv1_features, window_size) -> (batch_size, conv2_features, window_size)
            nn.Conv1d(in_channels=self.conv0_features, out_channels=self.conv1_features,
                      kernel_size=self.conv1_kernel_size, padding=(self.conv1_kernel_size // 2)),
            nn.ReLU(),

            # Average pooling: Halves the window size
            # Shape: (batch_size, conv1_features, window_size) -> (batch_size, conv1_features, window_size // 2)
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Shape: (batch_size, conv1_features, window_size) -> (batch_size, conv2_features, window_size)
            nn.Conv1d(in_channels=self.conv1_features, out_channels=self.conv2_features,
                      kernel_size=self.conv2_kernel_size, padding=(self.conv2_kernel_size // 2)),
            nn.ReLU(),

            # Shape: (batch_size, conv2_features, window_size) -> (batch_size, conv2_features * window_size)
            nn.Flatten(),

            # Shape: (batch_size, conv2_features * window_size) -> (batch_size, hidden1_features)
            nn.Linear(self.conv2_features * (window_size//2),
                      self.hidden1_features),
            nn.LayerNorm(self.hidden1_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            # Shape: (batch_size, hidden1_features) -> (batch_size, hidden2_features)
            nn.Linear(self.hidden1_features, self.hidden2_features),
            nn.LayerNorm(self.hidden2_features),
            nn.ReLU(),

            # Shape: (batch_size, hidden2_features) -> (batch_size, n_features)
            nn.Linear(self.hidden2_features, n_features)
        ]
        self.model = nn.Sequential(*self.layers)

        # Apply Glorot initialization to all layers
        if self.initialize_method == 'glorot':
            for layer in self.layers:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Apply He initialization to all layers
        if self.initialize_method == 'he':
            for layer in self.layers:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Change the shape to (batch_size, n_features, window_size)
        x = x.permute(0, 2, 1)
        return self.model(x)


class BaTriTempCNN1D(BaseModelWrapper):
    """
    CNN1D model for time series forecasting.
    """
    name = 'cnn1d_v3'

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

        # Model configurations
        self.config = kwargs.get('config', {})

        # Loss history
        self.losses = []

        # Get the available device
        self.device = get_available_device()

    def fit(self, generator, x, y):
        '''
        generator: WindowGenerator object
            data: (batch_size, window_size, n_features)
            target: (batch_size, n_features)
        '''
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
                target = torch.tensor(
                    target, dtype=torch.float32).to(self.device)

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

    def summary(self, config):
        # Print the model summary. The summary function is called before fit
        self.model = CNN1D(
            self.n_features, config['window_size'], self.config).to(self.device)

        print(f'{self.name} model summary:')
        print(self.model)

    def reset(self):
        del self.model
        clear_memory()

    def get_params(self):
        return {}
