import torch
import torch.nn as nn


class DQNet(nn.Module):
    def __init__(self, hiddenDim=128, hiddenLayers = 10):
        super(DQNet, self).__init__()

        self.firstLayer = nn.Linear(8, hiddenDim)
        self.forwardLayers = nn.ModuleList()
        self.lastLayer = nn.Linear(hiddenDim, 4)
        for i in range(hiddenLayers):
            self.forwardLayers.append(
                nn.Sequential(
                    nn.Linear(hiddenDim, hiddenDim),
                    nn.ReLU()
                )

            )

    def forward(self, x):
        x = self.firstLayer(x)
        for linearModule in self.forwardLayers:
            x = linearModule(x)
        return self.lastLayer(x)


if __name__ == '__main__':
    testNetwork = DQNet()
    print(testNetwork) # Prints network Architecture

    inputTensor = torch.rand(1, 8)
    output = testNetwork(inputTensor)
    print(output) # Forward pass through network

