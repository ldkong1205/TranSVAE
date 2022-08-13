import torch.nn.functional as F
import torch.nn as nn


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        self.c1 = dcgan_conv(nc, nf)
        self.c2 = dcgan_conv(nf, nf * 2)
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        self.c5 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder_convT(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_convT, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        self.upc4 = dcgan_upconv(nf * 2, nf)
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(
            input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


class decoder_woSkip(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_woSkip, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        self.upc4 = dcgan_upconv(nf * 2, nf)
        self.upc5 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(
            input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output


class upconv(nn.Module):
    def __init__(self, nc_in, nc_out):
        super().__init__()
        self.conv = nn.Conv2d(nc_in, nc_out, 3, 1, 1)
        self.norm = nn.BatchNorm2d(nc_out)

    def forward(self, input):
        out = F.interpolate(input, scale_factor=2,
                            mode='bilinear', align_corners=False)
        return F.relu(self.norm(self.conv(out)))


class decoder_conv(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_conv, self).__init__()
        self.dim = dim
        nf = 64

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
            upconv(nf * 8, nf * 4),
            upconv(nf * 4, nf * 2),
            upconv(nf * 2, nf * 2),
            upconv(nf * 2, nf),
            nn.Conv2d(nf, nc, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.dim, 1, 1))
        output = output.view(
            input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output