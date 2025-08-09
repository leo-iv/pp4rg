import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from dataclasses import dataclass

from .dataset import RobotConfigDataset


class Autoencoder(nn.Module):
    """
    An autoencoder neural network for compressing and reconstructing robot configurations
    using a learned latent representation.

    Args:
        conf_dim (int): The number of dimensions of the configuration of the robot.
        latent_dim (int): The number of dimensions of the latent space.
    """

    def __init__(self, conf_dim: int, latent_dim: int = 7):
        super(Autoencoder, self).__init__()

        self.input_dim = conf_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 384),
            nn.ReLU(),
            nn.Linear(384, 186),
            nn.ReLU(),
            nn.Linear(186, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 186),
            nn.ReLU(),
            nn.Linear(186, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def to_file(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        }, filepath)

    @classmethod
    def from_file(cls, filepath):
        data = torch.load(filepath)
        model = cls(data['input_dim'], data['latent_dim'])
        model.load_state_dict(data['model_state_dict'])
        return model


@dataclass
class TrainConfig:
    n_epochs: int = 150
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    train_val_split: float = 0.8
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model: Autoencoder, dataset: RobotConfigDataset, filepath: str = None, cfg: TrainConfig = TrainConfig()):
    """
    Trains an Autoencoder model on a dataset of robot configurations.

    Args:
        model (Autoencoder): The autoencoder model to train.
        dataset (RobotConfigDataset): The dataset containing robot configurations for training and validation.
        filepath (str): If provided, saves the best-performing model (based on validation loss) 
            to this file. Should end with `.pt`.
        cfg (TrainConfig): Training configuration parameters.
    """
    train_size = int(cfg.train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_vloss = float('inf')

    for epoch in range(cfg.n_epochs):
        print(f'EPOCH {epoch + 1}:')

        # training
        model.train()
        running_loss = 0.0
        for batch_idx, (input, target) in enumerate(train_dataloader):
            input, target = input.to(cfg.device), target.to(cfg.device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 1000 == 999:
                batch_loss = running_loss / 1000
                print(f'\tbatch: {batch_idx + 1}\tloss: {batch_loss:.5f}\tlr: {scheduler.get_last_lr()[0]}')
                running_loss = 0.0

        # validation
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for input, target in val_dataloader:
                input, target = input.to(cfg.device), target.to(cfg.device)
                output = model(input)
                vloss = criterion(output, target)
                running_vloss += vloss

        vloss = running_vloss / len(val_dataloader)
        scheduler.step(vloss)
        print(f'VALIDATION:\tvloss: {vloss:.5f}')

        # save best performing model
        if filepath is not None and vloss < best_vloss:
            best_vloss = vloss
            print(f"Saving model at {filepath}")
            model.to_file(filepath)

    print('Training completed')
