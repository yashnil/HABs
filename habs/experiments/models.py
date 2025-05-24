
import torch.nn as nn

class SimpleUNet(nn.Module):
    """Tiny U-Net: encoder–decoder with skip connections."""
    def __init__(self, in_ch=18, out_ch=1, base=32):
        super().__init__()
        C = base
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, C, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(C, C, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(C, 2*C, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(2*C, 2*C, 3, padding=1), nn.ReLU())
        self.bott = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(2*C, 4*C, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(4*C, 2*C, 2, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(4*C, 2*C, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(2*C, C, 2, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(2*C, C, 3, padding=1), nn.ReLU())
        self.out  = nn.Conv2d(C, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b  = self.bott(e2)
        d2 = self.dec2(torch.cat([b, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return self.out(d1).squeeze(1)          # (B,H,W)
