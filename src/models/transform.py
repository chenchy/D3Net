import torch
import torch.nn as nn

class BandSplit(nn.Module):
    def __init__(self, sections, dim=2):
        super().__init__()

        self.sections = sections
        self.dim = dim
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output: tuple of (batch_size, in_channels, sections[0], n_frames), ... (batch_size, in_channels, sections[-1], n_frames), where sum of sections is equal to n_bins
        """
        return torch.split(input, self.sections, dim=self.dim)
    
    def extra_repr(self):
        s = "1-{}, [".format(sum(self.sections))

        
        s += "1-{}".format(self.sections[0])
        start = self.sections[0] + 1

        for n_bins in self.sections[1:]:
            s += ", {}-{}".format(start, start + n_bins - 1)
            start += n_bins
        s += "]"

        return s

def _test_band_split():
    sections = [10, 20]
    batch_size, num_features, n_bins, n_frames = 2, 3, sum(sections), 5

    input = torch.randint(0, 10, (batch_size, num_features, n_bins, n_frames), dtype=torch.float)
    
    band_split = BandSplit(sections=sections)
    low, high = band_split(input)
    print(input.size(), low.size(), high.size())

if __name__ == '__main__':
    print("="*10, "BandSplit", "="*10)
    _test_band_split()
    print()