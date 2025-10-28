# models/puffin_D.py
import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with residual connection"""

    def __init__(self, inp, oup, expand_ratio=2):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class Puffin_D(nn.Module):
    """
    Puffin_D Model for high-performance transcription initiation prediction.
    Deep encoder-decoder network for 100kb sequences.
    """

    def __init__(self, config):
        super(Puffin_D, self).__init__()

        self.config = config
        model_config = config['model']

        # Encoder layers (upward pass)
        self.uplblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64)
            ),
            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
        ])

        # Encoder processing blocks
        self.upblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
        ])

        # Decoder layers (downward pass)
        self.downlblocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=17, padding=8),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64),
            ),
        ])

        # Decoder processing blocks
        self.downblocks = nn.ModuleList([
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
        ])

        # Second encoder pass
        self.uplblocks2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
        ])

        self.upblocks2 = nn.ModuleList([
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
        ])

        # Second decoder pass
        self.downlblocks2 = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(128, 128, kernel_size=17, padding=8),
                nn.BatchNorm1d(128),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(128, 96, kernel_size=17, padding=8),
                nn.BatchNorm1d(96),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(96, 64, kernel_size=17, padding=8),
                nn.BatchNorm1d(64),
            ),
        ])

        self.downblocks2 = nn.ModuleList([
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128)),
            nn.Sequential(ConvBlock(96, 96), ConvBlock(96, 96)),
            nn.Sequential(ConvBlock(64, 64), ConvBlock(64, 64)),
        ])

        # Final output layer - single target
        self.final = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),  # Single channel output
            nn.Softplus(),
        )

        self.use_cuda = model_config['use_cuda']
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        """Forward pass"""
        out = x
        encodings = []

        # First upward pass
        for lconv, conv in zip(self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        # First downward pass with skip connections
        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        # Second upward pass
        encodings3 = [out]
        for enc, lconv, conv in zip(reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        # Second downward pass
        for enc, lconv, conv in zip(reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        return out

    def predict(self, sequences):
        """Predict transcription initiation signals"""
        self.eval()
        predictions = []

        with torch.no_grad():
            for seq in sequences:
                if self.use_cuda:
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).cuda()
                else:
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0)

                pred = self(seq_tensor)
                predictions.append(pred.cpu().numpy())

        return predictions

    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']