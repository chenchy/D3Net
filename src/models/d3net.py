import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models.transform import BandSplit
from models.glu import GLU2d
from models.d2net import D2Block

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(
        self,
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[256,1344],
        scale=(2,2),
        num_d2blocks=None, depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        depth_final=None,
        eps=EPS,
        **kwargs
    ):
        super().__init__()

        self.bands, self.sections = bands, sections
        self.band_split = BandSplit(sections=sections, dim=2)

        out_channels = 0
        for band in bands:
            if out_channels < growth_rate[band][0] + growth_rate[band][-1]:
                out_channels = growth_rate[band][0] + growth_rate[band][-1]
        
        net = {}
        for band in bands:
            if growth_rate[band][0] + growth_rate[band][-1] < out_channels:
                _out_channels = out_channels
            else:
                _out_channels = None
            net[band] = D3NetBackbone(in_channels, num_features[band], growth_rate[band], kernel_size[band], scale=scale[band], num_d2blocks=num_d2blocks[band], depth=depth[band], out_channels=_out_channels, eps=eps)
        net['full'] = D3NetBackbone(in_channels, num_features['full'], growth_rate['full'], kernel_size['full'], scale=scale['full'], num_d2blocks=num_d2blocks['full'], depth=depth['full'], eps=eps)

        self.net = nn.ModuleDict(net)

        _in_channels = out_channels + growth_rate['full'][0] + growth_rate['full'][-1] # channels for 'low' & 'middle' + channels for 'full'
        
        if kernel_size_final is None:
            kernel_size_final = kernel_size

        self.d2block = D2Block(_in_channels, growth_rate_final, kernel_size_final, depth=depth_final, eps=eps)
        self.norm2d = nn.BatchNorm2d(growth_rate_final, eps=eps)
        self.glu2d = GLU2d(growth_rate_final, in_channels, kernel_size=(1,1), stride=(1,1))
        self.nonlinear2d = nn.ReLU()

        self.in_scale, self.in_bias = nn.Parameter(torch.Tensor(sum(sections),)), nn.Parameter(torch.Tensor(sum(sections),))
        self.out_scale, self.out_bias = nn.Parameter(torch.Tensor(sum(sections),)), nn.Parameter(torch.Tensor(sum(sections),))

        self.in_channels, self.num_features = in_channels, num_features
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.scale = scale
        self.num_d2blocks, self.depth = num_d2blocks, depth
        self.growth_rate_final = growth_rate_final
        self.kernel_size_final = kernel_size_final
        self.depth_final = depth_final
        self.eps = eps
        
        self._reset_parameters()

        self.num_parameters = self._get_num_parameters()
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
        Returns:
            output (batch_size, in_channels, n_bins, n_frames)
        """
        bands, sections = self.bands, self.sections
        n_bins = input.size(2)
        eps = self.eps

        sections = [sum(sections), n_bins - sum(sections)]
        x_valid, x_invalid = torch.split(input, sections, dim=2)

        x = (x_valid - self.in_bias.unsqueeze(dim=1)) / (self.in_scale.unsqueeze(dim=1) + eps)
        x = self.band_split(x)

        x_bands = []
        for band, x_band in zip(bands, x):
            x_band = self.net[band](x_band)
            x_bands.append(x_band)
        x_bands = torch.cat(x_bands, dim=2)

        x_full = self.net['full'](x_valid)
        x = torch.cat([x_bands, x_full], dim=1)

        x = self.d2block(x)
        x = self.norm2d(x)
        x = self.glu2d(x)
        x = self.nonlinear2d(x)
        x = self.out_scale.unsqueeze(dim=1) * x + self.out_bias.unsqueeze(dim=1)

        output = torch.cat([x, x_invalid], dim=2)

        return output
    
    def _reset_parameters(self):
        self.in_scale.data.fill_(1)
        self.in_bias.data.zero_()
        self.out_scale.data.fill_(1)
        self.out_bias.data.zero_()
    
    def get_package(self):
        config = {
            'in_channels': self.in_channels, 'num_features': self.num_features,
            'growth_rate': self.growth_rate,
            'kernel_size': self.kernel_size,
            'bands': self.bands, 'sections': self.sections,
            'scale': self.scale,
            'num_d2blocks': self.num_d2blocks, 'depth': self.depth,
            'growth_rate_final': self.growth_rate_final,
            'kernel_size_final': self.kernel_size_final,
            'depth_final': self.depth_final,
            'eps': self.eps
        }
        
        return config
    
    @classmethod
    def build_from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        in_channels = config['in_channels']
        bands = config['bands']

        sections = [
            config[band]['sections'] for band in bands
        ]
        num_features = {
            band: config[band]['num_features'] for band in bands + ['full']
        }
        growth_rate = {
            band: config[band]['growth_rate'] for band in bands + ['full']
        }
        kernel_size = {
            band: config[band]['kernel_size'] for band in bands + ['full']
        }
        scale = {
            band: config[band]['scale'] for band in bands + ['full']
        }
        num_d2blocks = {
            band: config[band]['num_d2blocks'] for band in bands + ['full']
        }
        depth = {
            band: config[band]['depth'] for band in bands + ['full']
        }

        growth_rate_final = config['final']['growth_rate']
        kernel_size_final = config['final']['kernel_size']
        depth_final = config['final']['depth']

        eps = config.get('eps') or EPS

        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            depth_final=depth_final,
            eps=eps
        )
        
        return model
    
    @classmethod
    def build_model(cls, model_path):
        config = torch.load(model_path, map_location=lambda storage, loc: storage)
    
        in_channels, num_features = config['in_channels'], config['num_features']
        growth_rate = config['growth_rate']

        kernel_size = config['kernel_size']
        bands, sections = config['bands'], config['sections']
        scale = config['scale']

        num_d2blocks, depth = config['num_d2blocks'], config['depth']

        growth_rate_final = config['growth_rate_final']
        kernel_size_final = config['kernel_size_final']
        depth_final = config['depth_final']

        eps = config['eps']
        
        model = cls(
            in_channels, num_features,
            growth_rate,
            kernel_size,
            bands=bands, sections=sections,
            scale=scale,
            num_d2blocks=num_d2blocks, depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            depth_final=depth_final,
            eps=eps
        )
        
        return model
    
    def _get_num_parameters(self):
        num_parameters = 0
        
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
                
        return num_parameters

class D3NetBackbone(nn.Module):
    def __init__(self, in_channels, num_features, growth_rate, kernel_size, scale=(2,2), num_d2blocks=None, depth=None, out_channels=None, eps=EPS):
        """
        Args:
            in_channels <int>
            num_features <int>
            growth_rate <list<int>>: `len(growth_rate)` must be an odd number.
            kernel_size <int>
            scale <int> or <list<int>>: Upsampling and Downsampling scale
        """
        super().__init__()

        assert len(growth_rate) % 2 == 1, "`len(growth_rate)` must be an odd number."

        kernel_size = _pair(kernel_size)
        num_encoder_blocks = len(growth_rate) // 2

        # Network
        self.conv2d = nn.Conv2d(in_channels, num_features, kernel_size, stride=(1,1))

        encoder, decoder = [], []
        encoder = Encoder(num_features, growth_rate[:num_encoder_blocks], kernel_size=kernel_size, down_scale=scale, num_d2blocks=num_d2blocks[:num_encoder_blocks], depth=depth[:num_encoder_blocks], eps=eps)

        skip_channels = []
        for downsample_block in encoder.net:
            skip_channels.append(downsample_block.out_channels)        
        skip_channels = skip_channels[::-1]

        # encoder.net[-1].out_channels == skip_channels[0]
        _in_channels, _growth_rate = encoder.net[-1].out_channels, growth_rate[num_encoder_blocks]
        bottleneck_d3block = D3Block(_in_channels, _growth_rate, kernel_size=kernel_size, num_blocks=num_d2blocks[num_encoder_blocks], depth=depth[num_encoder_blocks])

        _in_channels = bottleneck_d3block.out_channels
        decoder = Decoder(_in_channels, skip_channels, growth_rate[num_encoder_blocks+1:], kernel_size=kernel_size, up_scale=scale, num_d2blocks=num_d2blocks[num_encoder_blocks+1:], depth=depth[num_encoder_blocks+1:], eps=eps)
        
        self.encoder = encoder
        self.bottleneck_conv2d = bottleneck_d3block
        self.decoder = decoder

        if out_channels is not None:
            _in_channels = decoder.out_channels

            net = []
            net.append(nn.BatchNorm2d(_in_channels, eps=eps))
            net.append(nn.Conv2d(_in_channels, out_channels, kernel_size=(1,1), stride=(1,1)))

            self.pointwise_conv2d = nn.Sequential(*net)
        else:
            self.pointwise_conv2d = None

        self.kernel_size = kernel_size
        self.out_channels = out_channels
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        Ph, Pw = Kh - 1, Kw - 1
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        x = self.conv2d(input)
        x, skip = self.encoder(x)
        x = self.bottleneck_conv2d(x)
        x = self.decoder(x, skip[::-1])

        if self.pointwise_conv2d:
            output = self.pointwise_conv2d(x)
        else:
            output = x
        
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, down_scale=(2,2), num_d2blocks=None, depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")
        
        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")
            
        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            downsample_block = DownSampleD3Block(_in_channels, growth_rate[idx], kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_d2blocks[idx], depth=depth[idx], eps=eps)
            net.append(downsample_block)
            _in_channels = downsample_block.out_channels
        
        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks
    
    def forward(self, input):
        num_d3blocks = self.num_d3blocks

        x = input
        skip = []

        for idx in range(num_d3blocks):
            x, x_skip = self.net[idx](x)
            skip.append(x_skip)
        
        output = x

        return output, skip

class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, growth_rate, kernel_size, up_scale=(2,2), num_d2blocks=None, depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: 
            skip_channels <list<int>>:
            growth_rate <list<int>>:
            kernel_size <tuple<int>> or <int>:
            num_d2blocks <list<int>> or <int>:
        """
        super().__init__()

        if type(growth_rate) is list:
            num_d3blocks = len(growth_rate)
        else:
            # TODO: implement
            raise ValueError("`growth_rate` must be list.")
        
        if num_d2blocks is None:
            num_d2blocks = [None] * num_d3blocks
        elif type(num_d2blocks) is int:
            num_d2blocks = [num_d2blocks] * num_d3blocks
        elif type(num_d2blocks) is list:
            assert num_d3blocks == len(num_d2blocks), "Invalid length of `num_d2blocks`"
        else:
            raise ValueError("Invalid type of `num_d2blocks`.")
            
        if depth is None:
            depth = [None] * num_d3blocks
        elif type(depth) is int:
            depth = [depth] * num_d3blocks
        elif type(depth) is list:
            assert num_d3blocks == len(depth), "Invalid length of `depth`"
        else:
            raise ValueError("Invalid type of `depth`.")

        num_d3blocks = len(growth_rate)
        net = []

        _in_channels = in_channels

        for idx in range(num_d3blocks):
            upsample_block = UpSampleBlock(_in_channels, growth_rate[idx], kernel_size=kernel_size, up_scale=up_scale, num_blocks=num_d2blocks[idx], depth=depth[idx], eps=eps)
            net.append(upsample_block)
            _in_channels = upsample_block.out_channels + skip_channels[idx]
        
        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks
        self.out_channels = _in_channels
    
    def forward(self, input, skip):
        num_d3blocks = self.num_d3blocks

        x = input

        for idx in range(num_d3blocks):
            x_skip = skip[idx]
            x = self.net[idx](x)
            
            """
            _, _, H, W = x.size()
            _, _, H_skip, W_skip = x_skip.size()
            padding_height = H - H_skip
            padding_width = W - W_skip
            padding_top = padding_height//2
            padding_bottom = padding_height - padding_top
            padding_left = padding_width//2
            padding_right = padding_width - padding_left

            x = F.pad(x, (-padding_left, -padding_right, -padding_top, -padding_bottom))
            """

            x = torch.cat([x, x_skip], dim=1)
        
        output = x

        return output

class D3Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), num_blocks=None, depth=None, eps=EPS):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels, TODO: <list<list<int>>>
            kernel_size <int> or <tuple<int>>: Kernel size
            num_blocks <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `num_blocks`.
            depth <int>: 
        """
        super().__init__()

        if type(growth_rate) is int:
            assert num_blocks is not None, "Specify `num_blocks`"
            growth_rate = [growth_rate] * num_blocks
        elif type(growth_rate) is list:
            if num_blocks is not None:
                assert num_blocks == len(growth_rate), "`num_blocks` is different from `len(growth_rate)`"
            num_blocks = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
    
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks
        self.out_channels = growth_rate[-1]

        net = []
        _in_channels = in_channels

        for idx in range(num_blocks):
            _out_channels = sum(growth_rate[idx:])
            d2block = D2Block(_in_channels, _out_channels, kernel_size=kernel_size, depth=depth, eps=eps)
            net.append(d2block)
            _in_channels = growth_rate[idx]

        self.net = nn.ModuleList(net)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by ... 
        """
        growth_rate, num_blocks = self.growth_rate, self.num_blocks

        x = input
        x_residual = 0

        for idx in range(num_blocks):
            x = self.net[idx](x)
            x_residual = x_residual + x
            
            in_channels = growth_rate[idx]
            stacked_channels = sum(growth_rate[idx+1:])
            sections = [in_channels, stacked_channels]

            if idx != num_blocks - 1:
                x, x_residual = torch.split(x_residual, sections, dim=1)
        
        output = x_residual

        return output

class DownSampleD3Block(nn.Module):
    """
    D3Block + down sample
    """
    def __init__(self, in_channels, growth_rate, kernel_size=(3,3), down_scale=(2,2), num_blocks=None, depth=None, eps=EPS):
        super().__init__()

        self.down_scale = _pair(down_scale)

        self.d3block = D3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, depth=depth, eps=eps)
        self.downsample2d = nn.AvgPool2d(kernel_size=self.down_scale, stride=self.down_scale)

        self.out_channels = self.d3block.out_channels
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output:
                (batch_size, growth_rate[-1], H_down, W_down) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H_down, W_down) if type(growth_rate) is int
                where H_down = H // down_scale[0] and W_down = W // down_scale[1]
            skip:
                (batch_size, growth_rate[-1], H, W) if type(growth_rate) is list<int>
                or (batch_size, growth_rate, H, W) if type(growth_rate) is int
        """
        _, _, n_bins, n_frames = input.size()

        Kh, Kw = self.down_scale
        Ph, Pw = (Kh - n_bins % Kh) % Kh, (Kw - n_frames % Kw) % Kw
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        input = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        
        x = self.d3block(input)
        skip = x

        output = self.downsample2d(x)

        return output, skip

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=(2,2), up_scale=(2,2), num_blocks=None, depth=None, eps=EPS):
        super().__init__()

        self.norm2d = nn.BatchNorm2d(in_channels, eps=eps)
        self.upsample2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=up_scale, stride=up_scale)
        self.d3block = D3Block(in_channels, growth_rate, kernel_size, num_blocks=num_blocks, depth=depth, eps=eps)

        self.out_channels = self.d3block.out_channels
    
    def forward(self, input):
        # TODO: chomp
        x = self.norm2d(input)
        x = self.upsample2d(x)
        output = self.d3block(x)

        return output

def _test_d3block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    num_blocks, depth = 2, 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Block(in_channels, growth_rate, kernel_size=kernel_size, num_blocks=num_blocks, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size(), model.out_channels)
    print()

    growth_rate = [3, 4, 5, 6]
    model = D3Block(in_channels, growth_rate, kernel_size=kernel_size, depth=depth)

    print(model)
    output = model(input)
    print(input.size(), output.size(), model.out_channels)

def _test_down_d3block():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 3
    growth_rate = 2
    kernel_size = (3, 3)
    down_scale = (2, 2)
    num_blocks, depth = 2, 4

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = DownSampleD3Block(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, num_blocks=num_blocks, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size(), model.out_channels)
    print()

    growth_rate = [3, 4, 5, 6]
    model = DownSampleD3Block(in_channels, growth_rate, kernel_size=kernel_size, down_scale=down_scale, depth=depth)

    print(model)
    output, skip = model(input)
    print(input.size(), output.size(), skip.size(), model.out_channels)

def _test_encoder():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels = 32

    growth_rate = [2, 3, 4]
    kernel_size = 3
    num_d2blocks = 2
    
    depth = [2, 2, 3]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())
    for _skip in skip:
        print(_skip.size())
    print()

    depth = 2
    encoder = Encoder(in_channels, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)
    output, skip = encoder(input)

    print(encoder)
    print(input.size(), output.size())

    for _skip in skip:
        print(_skip.size())
    print()

def _test_d3net_backbone():
    batch_size = 4
    n_bins, n_frames = 16, 64
    in_channels, num_features = 2, 32

    growth_rate = [2, 3, 4, 3, 2]
    kernel_size = 3
    num_d2blocks = [2, 2, 2, 2, 2]
    
    depth = [3, 3, 4, 2, 2]
    input = torch.randn(batch_size, in_channels, n_bins, n_frames)

    model = D3NetBackbone(in_channels, num_features, growth_rate, kernel_size, num_d2blocks=num_d2blocks, depth=depth)

    print(model)

    output = model(input)

    print(input.size(), output.size())

def _test_d3net():
    config_path = "./data/d3net/vocals_toy.yaml"
    batch_size, in_channels, n_bins, n_frames = 4, 2, 257, 128 # 4, 2, 2049, 256

    input = torch.randn(batch_size, in_channels, n_bins, n_frames)
    model = D3Net.build_from_config(config_path)
    
    output = model(input)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    print('='*10, "D3Block", '='*10)
    _test_d3block()
    print()

    print('='*10, "DownSampleD3Block", '='*10)
    _test_down_d3block()
    print()

    print('='*10, "Encoder", '='*10)
    _test_encoder()
    print()

    print('='*10, "D3Net backbone", '='*10)
    _test_d3net_backbone()
    print()

    print('='*10, "D3Net", '='*10)
    _test_d3net()