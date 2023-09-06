import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

URLS = {
    "hubert-discrete": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-discrete-d49e1c77.pt",
    "hubert-soft": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-0321fd7e.pt",
}

class AcousticModel(nn.Module):
    def __init__(self, content_dim, content_outdim, timbre_dim, timbre_outdim, decoder_hidden, out_dim, upsample: bool = True):
        super().__init__()
        self.content_encoder = Encoder(content_dim, 256, content_outdim, upsample)
        self.timbre_encoder = Encoder(timbre_dim, 256, timbre_outdim, upsample)
        self.decoder = Decoder(decoder_hidden, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.content_encoder(x)
        y = self.timbre_encoder(y)
        return self.decoder(x, y)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, upsample: bool = True):
        super().__init__()
        self.prenet = PreNet(input_dim, hidden_dim, hidden_dim)
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(output_dim),
            nn.ConvTranspose1d(output_dim, output_dim, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(output_dim, output_dim, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(output_dim),
            nn.Conv1d(output_dim, output_dim, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        x  = torch.cat((x, mels), dim=-1)
        x, _ = self.lstm1(x)
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(xs.size(0), out_dim, device=xs.device)
        h1 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)
        c1 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)
        h2 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)
        c2 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)
        h3 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)
        c3 = torch.zeros(1, xs.size(0), hidden_dim, device=xs.device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


class PreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _acoustic(
    name: str,
    discrete: bool,
    upsample: bool,
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    acoustic = AcousticModel(discrete, upsample)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(URLS[name], progress=progress)
        consume_prefix_in_state_dict_if_present(checkpoint["acoustic-model"], "module.")
        acoustic.load_state_dict(checkpoint["acoustic-model"])
        acoustic.eval()
    return acoustic


def hubert_discrete(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Discrete acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-discrete",
        discrete=True,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )


def hubert_soft(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Soft acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-soft",
        discrete=False,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )