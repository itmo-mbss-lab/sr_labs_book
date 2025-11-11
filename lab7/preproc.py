# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script performes preprocessing procedures


# Import of modules
import torch
import torch.nn.functional as F


class PreEmphasis(torch.nn.Module):
    # Preemphasis procedure

    def __init__(self, coef: float = 0.97):
        
        super().__init__()
        
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        
        return F.conv1d(input, self.flipped_filter).squeeze(1)