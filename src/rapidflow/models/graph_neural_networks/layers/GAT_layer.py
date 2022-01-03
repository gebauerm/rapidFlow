import torch.nn as nn
import torch


class GATLayer(nn.Module):

    def __init__(self, size_in, size_out, heads=1, mean=None):
        """
        Implementation of a GAT-Layer as in: https://arxiv.org/pdf/1710.10903.pdf
        The class assumes, that the neighborhood is already passed to the model.

        Args:
            size_in ([type]): [description]
            size_out ([type]): [description]
            heads (int, optional): [description]. Defaults to 1.
            mean ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.mean = mean
        self.heads = heads
        self.attention_heads = nn.ModuleList(
            [AttentionMechanism(self.size_in, self.size_out) for _ in range(self.heads)])

    def forward(self, x, n_x):
        output = torch.empty((x.shape[0], self.size_out, self.heads))
        for idx, head in enumerate(self.attention_heads):
            output[:, :, idx] = head(x, n_x).t()  # bug
        if self.mean:
            output = output.mean(dim=2)
        return output


class AttentionMechanism(nn.Module):
    def __init__(self, size_in, size_out):
        """
        A single Attention Head for the GAT-Layer. Weights are initialized with Xavier Uniform.
        Attention implementation based on: https://arxiv.org/pdf/1710.10903.pdf

        Args:
            size_in ([type]): [description]
            size_out ([type]): [description]
        """
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        W = torch.empty(size_out, size_in)  # W in R^{F'x F}
        self.W = nn.Parameter(W)
        a = torch.empty(2*size_out,  1)
        self.a = nn.Parameter(a)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # initialize
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, n_x):
        alpha = self._compute_alpha(x, n_x)
        x = alpha.matmul(n_x).t()
        x = self.W.matmul(x)
        return self.sigmoid(x)

    def _compute_alpha(self, x, n_x):
        x = x.view(self.size_in, -1)
        n_x = n_x.view(self.size_in, -1)
        x = self.W.matmul(x)  # I could spare this one, but have to reshape my matrices than
        n_x = self.W.matmul(n_x)
        energy = self.a.t().matmul(torch.cat((x.repeat(1, n_x.shape[1]), n_x)))
        return self.softmax(energy)


if __name__ == "__main__":
    a = torch.randn(1, 10)
    a_n = torch.randn(5, 10)
    layer = GATLayer(10, 5, heads=3)
    print(layer(a, a_n))
