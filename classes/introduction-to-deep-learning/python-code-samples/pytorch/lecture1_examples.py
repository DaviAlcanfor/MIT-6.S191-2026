import torch
from torch import nn


class MyDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))

    def forward(self, inputs):
        z = torch.matmul(inputs, self.W) + self.b
        output = torch.sigmoid(z)
        return output


layer = nn.Linear(in_features=m, out_features=2)


model_two_layers = nn.Sequential(
    nn.Linear(m, n),
    nn.ReLU(),
    nn.Linear(n, 2)
)


model_deep = nn.Sequential(
    nn.Linear(m, n1),
    nn.ReLU(),
    nn.Linear(n1, n2),
    nn.ReLU(),
    nn.Linear(n2, 2)
)


weights = torch.randn(..., requires_grad=True)

while True:
    loss = compute_loss(weights)
    loss.backward()
    weights = weights - lr * weights.grad
