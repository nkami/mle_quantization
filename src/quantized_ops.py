import torch.nn as nn
import torch


def quantize_model(model, w_bits, a_bits):
    d = vars(model)['_modules']
    for key, val in d.items():
        if len(vars(val)['_modules']) > 0:
            quantize_model(val, w_bits, a_bits)
        elif isinstance(val, nn.Linear):
            d[key] = FCQuantized(val, w_bits, a_bits)
            pass
        elif isinstance(val, nn.Conv2d):
            d[key] = Conv2dQuantized(val, w_bits, a_bits)
            pass


class QuantizedModule(nn.Module):
    def __init__(self, w_bits: int = 4, a_bits: int = 4):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits

    # asymmetric quantization with zero conserved
    def quantize(self, x: torch.Tensor, n_bits: int = 4):
        eps = 1e-5
        region = torch.max(x) - torch.min(x) + eps
        qx = torch.tensor([(2 ** n_bits - 1) / region]).float().to(x.device)
        zpx = torch.round(qx * torch.min(x)).to(x.device)
        xq = torch.round(qx * x - zpx).float().to(x.device)
        return ((xq + zpx) / qx).to(x.device)

    def non_uniform_quantize(self, x: torch.Tensor, n_bits: int = 4):
        eps = 1e-5
        j = 1  # j = 0 -> width 8, j = 1 -> width 4, j = 2 -> width 2
        region = torch.max(x) - torch.min(x) + eps
        normalized_max = (2 ** (n_bits - 1 - j)) * (((2 ** n_bits - 1) // 2) + 2 ** j)
        normalized_x = ((x - torch.min(x)) / region) * normalized_max
        split_val = 2 ** (n_bits - 1)
        sparse_bin_width = split_val * 2 ** (-j)
        sparse_mask = (normalized_x > split_val).to(x.device)
        dense_mask = (normalized_x <= split_val).to(x.device)
        dense_bins = torch.round(normalized_x).to(x.device) * dense_mask
        sparse_bins = torch.round(normalized_x / sparse_bin_width).to(x.device) * sparse_bin_width * sparse_mask
        xq = dense_bins + sparse_bins
        qx = normalized_max / region
        return ((xq / qx) + torch.min(x)).to(x.device)

    def quantize_acts_per_channel(self, x: torch.Tensor, n_bits: int = 4):
        channels = []
        for idx in range(x.shape[1]):
            cq = self.quantize(x[:, idx, :, :], n_bits)
            channels.append(cq)
        return torch.stack(channels, dim=1).float().to(x.device)

    def quantize_weights_per_channel(self, x: torch.Tensor, n_bits: int = 4):
        channels = []
        for idx in range(x.shape[0]):
            cq = self.quantize(x[idx, :, :, :], n_bits)
            channels.append(cq)
        return torch.stack(channels, dim=0).float().to(x.device)


class FCQuantized(QuantizedModule):
    def __init__(self, fc: nn.Linear, w_bits: int = 4, a_bits: int = 4):
        super(FCQuantized, self).__init__(w_bits, a_bits)
        self.fc = fc
        with torch.no_grad():
            wq = self.quantize(self.fc.weight, n_bits=self.w_bits)
            self.fc.weight.copy_(wq.to(self.fc.weight.device))
            if self.fc.bias is not None:
                bq = self.quantize(self.fc.bias, n_bits=self.w_bits)
                self.fc.bias.copy_(bq.to(self.fc.weight.device))

    def forward(self, xf):
        if torch.sum(xf < 0) > 0:
            xq = self.quantize(xf, n_bits=self.a_bits)
        else:
            xq = self.non_uniform_quantize(xf, n_bits=self.a_bits)
        return self.fc.forward(xq)


class Conv2dQuantized(QuantizedModule):
    def __init__(self, conv: nn.Conv2d, w_bits: int = 4, a_bits: int = 4):
        super(Conv2dQuantized, self).__init__(w_bits, a_bits)
        self.conv = conv
        with torch.no_grad():
            # wq = self.quantize_weights_per_channel(self.conv.weight, n_bits=self.w_bits, clip=clip_weights)
            wq = self.quantize(self.conv.weight, n_bits=self.w_bits)
            self.conv.weight.copy_(wq)
            if conv.bias is not None:
                bq = self.quantize(conv.bias, n_bits=self.w_bits)
                self.conv.bias.copy_(bq)

    def forward(self, xf):
        if torch.sum(xf < 0) > 0:
            xq = self.quantize(xf, n_bits=self.a_bits)
        else:
            xq = self.non_uniform_quantize(xf, n_bits=self.a_bits)
        return self.conv.forward(xq)
