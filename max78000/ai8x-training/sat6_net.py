import torch
import torch.nn as nn
import ai8x


class SAT6Net(nn.Module):
    def __init__(self, num_classes=6, num_channels=4, dimensions=(28, 28), bias=False, **kwargs):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1, bias=bias, **kwargs)
        self.pool1 = ai8x.MaxPool2d(2, 2, **kwargs)

        self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, padding=1, bias=bias, **kwargs)
        self.pool2 = ai8x.MaxPool2d(2, 2, **kwargs)

        # Adicionamos um Pooling aqui para comprimir os dados para o hardware
        self.conv3 = ai8x.FusedConv2dReLU(32, 64, 3, padding=1, bias=bias, **kwargs)
        self.pool3 = ai8x.MaxPool2d(2, 2, **kwargs)

        # Novo tamanho matemático: 64 canais * 3 * 3 = 576 (menor que o limite de 1024 da placa)
        self.fc = ai8x.Linear(64 * 3 * 3, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)  # Chamando a nova camada no fluxo
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def sat6_net(pretrained=False, **kwargs):
    assert not pretrained
    return SAT6Net(**kwargs)


# Crachá de identificação do modelo
models = [
    {
        'name': 'sat6_net',
        'min_input': 1,
        'dim': 2,
        'input': (4, 28, 28),
    }
]