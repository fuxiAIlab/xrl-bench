import torch.nn as nn
import torch.nn.functional as F
import torch


class QNetwork(nn.Module):
    def __init__(self, action_size):
        """
        Deep Q-Network (DQN) model for reinforcement learning.

        Parameters:
        -----------
        action_size : int
            Number of possible actions.

        Attributes:
        -----------
        device : torch.device
            The device used for computations.
        conv1 : nn.Conv2d
            The first convolutional layer.
        conv2 : nn.Conv2d
            The second convolutional layer.
        conv3 : nn.Conv2d
            The third convolutional layer.
        fc1 : nn.Linear
            The first fully connected layer.
        fc2 : nn.Linear
            The second fully connected layer.
        """
        super(QNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def init_weights(self, m):
        """
        Initialize the weights of the model.

        Parameters:
        -----------
        m : nn.Module
            The module to initialize the weights for.
        """
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through network.

        Parameters:
        -----------
        torch.Tensor
            The output of the model.
        """
        # x = x.to(self.device).float()/255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x