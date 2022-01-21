import torch.nn as nn
import torch
from torch.optim import Adam
from rapidflow.nn_model import NNModel


class MLP(NNModel):
    def __init__(self, input_size, hidden_layer_config, output_size, learning_rate, weight_decay):
        """
        A simple Feed Forward Network with a Sigmoid Activation Function after the final layer.
        Performs binary classification and thus uses binary cross entropy loss.

        Args:
            input_size ([type]): [description]
            hidden_layer_config ([type]): [description]
            output_size ([type]): [description]
            learning_rate ([type]): [description]
            weight_decay ([type]): [description]
        """
        super().__init__(lr=learning_rate, weight_decay=weight_decay)
        self.layers = self._construct_layer(
            input_size=input_size, hidden_layer_config=hidden_layer_config, output_size=output_size)
        self.train_criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def _construct_layer(self, input_size, hidden_layer_config, output_size):
        """
        Generic Layer construction.

        Args:
            input_size ([type]): [description]
            hidden_layer_config ([type]): [description]
            output_size ([type]): [description]

        Returns:
            [type]: [description]
        """
        layers = nn.ModuleList([])
        for hidden_layer_size in hidden_layer_config:
            layers.append(nn.Linear(input_size, hidden_layer_size))
            input_size = hidden_layer_size
        layers.append(nn.Linear(input_size, output_size))
        return layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def loss_function(self, x, target, criterion):
        return criterion(x.flatten(), target)

    def predict(self, logits):
        probabilities = self.sigmoid(logits)
        predictions = probabilities.to(dtype=torch.int16).flatten()
        return predictions

    def train_step(self, x, y):
        logits = self(x)
        loss = self.loss_function(logits, y, self.train_criterion)
        loss.backward()
        self.optimizer.step()
        return logits

    def test_step(self, x):
        logits = self(x)
        predictions = self.predict(logits)
        return predictions

    def create_optimizer(self):
        """Needs to be defined in order to set the optimizer.
        """
        self.optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
