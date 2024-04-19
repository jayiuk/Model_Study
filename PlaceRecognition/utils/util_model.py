import torch.nn as nn

class EmbedNet(nn.Module):
    def __init__(self, backbone, model):
        super(EmbedNet, self).__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, x):
        x = self.backbone(x)
        embedded_x = self.model(x)
        return embedded_x

class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)