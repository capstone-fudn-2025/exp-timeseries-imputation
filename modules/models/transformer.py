"""
Transformer model for time series forecasting.
"""
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from ..utils.utils import get_available_device, forecast_support_torch, clear_memory
from ._base import BaseModelWrapper


class TransformerModel(nn.Module):
    def __init__(self, n_features: int, config: dict):
        super(TransformerModel, self).__init__()

        self.n_features = n_features

        self.num_encoder_layers = config.get('num_encoder_layers', 4)
        self.num_decoder_layers = config.get('num_decoder_layers', 4)
        self.dim_feedforward = config.get('dim_feedforward', 256)

        self.transformer = nn.Transformer(
            d_model=n_features,
            nhead=n_features,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tgt = torch.ones((x.shape[0], 1, self.n_features)).to(x.device)
        return self.transformer(x, tgt)


class TransformerTS(BaseModelWrapper):
    """
    Recurrent Network model for time series forecasting.
    """
    name = 'Transformer'

    def __init__(self, n_features: int = 1, layers=None, **kwargs):
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

        # Train loop on epochs
        for epoch in range(self.epochs):
            # Train loop on batches
            _datagen = tqdm(generator, desc=f'Epoch {epoch+1}/{self.epochs}')
            for data, target in _datagen:
                # Convert the numpy array to tensor. Repeat the data for the transformer
                data = torch.tensor(data, dtype=torch.float32).to(self.device)
                target = torch.tensor(
                    target, dtype=torch.float32).to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output[:, :, 0], target)

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
        # If `is_generator` is True, the system will give generator as `WindowGenerator`.
        # x is None
        # generator is a `WindowGenerator` object

        # If `is_generator` is False, the system will give x as numpy arrays.
        # generator is None
        # x is a numpy array of shape (data_length, window_size, n_features)
        raise NotImplementedError(
            'Prediction is currently not supported for TransformerTS model.')

    def forecast(self, x, steps):
        '''
        x: np.array
            Shape: (window_size, n_features)
        steps: int
        '''
        # Convert the numpy array to tensor
        x_tensor = torch.tensor(
            x.copy(), dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forecast the output
        return forecast_support_torch(self.model, x_tensor, steps, post_func=lambda x: x[:, :, 0])

    def summary(self):
        # Print the model summary. The summary function is called before fit
        self.model = TransformerModel(
            self.n_features, self.config).to(self.device)

        print(f'{self.name} model summary:')
        print(self.model)

    def reset(self):
        del self.model
        clear_memory()

    def get_params(self):
        return {}
