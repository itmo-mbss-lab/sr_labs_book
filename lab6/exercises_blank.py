# Exercises in order to perform laboratory work


# Import of modules
import os
import random
from enum import Enum
from typing import Iterable, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import Dataset

import whisper
from whisper.model import (
    Conv1d,
    ResidualAttentionBlock,
    LayerNorm,
    sinusoids,
)

from common import loadWAV, AugmentWAV
from whisper_fbanks import ExtractWhisperFbanks80


class train_dataset_loader(Dataset):
    # Train dataset loader
    
    def __init__(self, train_list, max_frames, train_path, augment=False, musan_path=None, rir_path=None):

        self.max_frames  = max_frames
        self.augment     = augment

        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        dictkeys = list(set([x.split()[0] for x in train_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

 
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(train_list):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):
        
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)

        if self.augment:
            ###########################################################
            # Here is your code

            pass
            
            ###########################################################

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)
    
class test_dataset_loader(Dataset):
    # Test dataset loader
    
    def __init__(self, test_list, max_frames, test_path):

        self.max_frames  = max_frames

        dictkeys = list(set([x.split()[0] for x in test_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(test_list):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(test_path, data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, index):

        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=1)

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        
        return len(self.data_list)

class WhisperEncoder(nn.Module):
    # Whisper frame level block
    
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = x.squeeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + self.positional_embedding[: x.shape[1], ...]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)

        x = x.permute((0, 2, 1))
        return x

    @staticmethod
    def build_model(
        model_name: str,
        load_weight=True,
        n_layer: int = 0,
    ):

        w = whisper.load_model(model_name, device="cpu")
        dims = w.dims

        model = WhisperEncoder(
             n_mels=dims.n_mels,
             n_ctx=dims.n_audio_ctx,
             n_state=dims.n_audio_state,
             n_head=dims.n_audio_head,
             n_layer=dims.n_audio_layer + n_layer,
         )
        
        if load_weight:
             full_sd = w.state_dict()
             encoder_sd = {
                 key.replace("encoder.", "", 1): val
                 for key, val in full_sd.items()
                 if key.startswith("encoder.")
             }
             model.load_state_dict(encoder_sd, strict=False)

        model.train()
        model.requires_grad_(True)

        return model
    
class TDNNSimple(nn.Module):
    # TDNN matching layer between framel level block and top-level layers
    
    def __init__(self, n_layers=1, in_ch=768, out_ch=1024):
        super().__init__()
        if n_layers > 1:
            self.model = nn.Sequential(
                *(
                    [
                        nn.Conv1d(
                            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True
                        ),
                        nn.GELU(),
                    ]
                    + sum(
                        [
                            [
                                nn.Conv1d(
                                    out_ch,
                                    out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True,
                                ),
                                nn.GELU(),
                            ]
                            for _ in range(n_layers - 1)
                        ]
                    )
                )
            )
        else:
            self.model = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv1d(
                    out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )

    def forward(self, x):
        x = self.model(x)
        return x

class StatPoolMode(Enum):
    # Statistics pooling mode
    
    M = 0
    V = 1
    MV = 2

class StatPoolLayer(nn.Module):
    # Statisctics pooling layer

    @classmethod
    def by_string_mode(cls, mode: str):
        mode = StatPoolMode[mode]
        return cls(mode)

    def __init__(self, stat_pool_mode: StatPoolMode, dim=-1):
        super(StatPoolLayer, self).__init__()

        self.mode = stat_pool_mode
        self.dim = dim

    def forward(self, x, **kwargs):
        mean_x = x.mean(self.dim)
        mean_x2 = x.pow(2).mean(self.dim)

        std_x = F.relu(mean_x2 - mean_x.pow(2)).sqrt()
        
        if self.mode == StatPoolMode.M:
            out = mean_x
        elif self.mode == StatPoolMode.V:
            out = std_x
        elif self.mode == StatPoolMode.MV:
            out = torch.cat([mean_x, std_x], dim=-1)
        else:
            raise ValueError('Operation\'s mode is incorrect')
        out = torch.flatten(out, 1)
        
        return out

class MaxoutLinear(nn.Module):
    # Maxout linear layer
    
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))

class MaxoutSegmentLevel(nn.Module):
    # Segment level block

    def __init__(self, input_dim: Union[int, List[int]], output_dim: Union[int, List[int]], enable_batch_norm: bool, fixed: bool = False):
        super(MaxoutSegmentLevel, self).__init__()

        if isinstance(input_dim, int) and isinstance(output_dim, int):
            input_dim = [input_dim]
            output_dim = [output_dim]
        self.num_layers = len(input_dim)
        self.enable_batch_norm = enable_batch_norm
        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.fixed = fixed

        for idx in range(self.num_layers):
            self.layers.append(MaxoutLinear(input_dim[idx], output_dim[idx]))

            if self.enable_batch_norm:
                self.bn.append(nn.BatchNorm1d(output_dim[idx], affine=False))

    def forward(self, x, **kwargs):

        for idx in range(self.num_layers):
            x = self.layers[idx](x)

            if self.enable_batch_norm:
                x = self.bn[idx](x)

        return x
    
class SSLDownstream(nn.Module):
    # Full speaker embedding extractor

    def __init__(
        self,
        preproc: ExtractWhisperFbanks80,
        feat_extractor: nn.Module,
        frame_level_block: nn.Module,
        pooling_block: nn.Module,
        segment_level_block: nn.Module,
        freeze_feats:bool = True
    ):
        super().__init__()
        self.preproc = preproc
        self.feat_extractor = feat_extractor
        self.frame_level =  frame_level_block
        self.pooling_level = pooling_block
        self.segment_level = segment_level_block
        self.freeze_feats = freeze_feats

    def forward(self, x: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, 
                add_labels: Optional[torch.Tensor] = None,
                lambda_f: Optional[torch.Tensor] = None,
                duration:float = 4):
        
        x = self.preproc(x)

        if self.freeze_feats:
            with torch.no_grad():
                self.feat_extractor.eval()
                x = self.feat_extractor(x)
        else:
            x = self.feat_extractor(x)
            
        ###########################################################
        # Here is your code

        ###########################################################
        
        return x

class MainModel(nn.Module):
    # Full computational graph including the neural network model and the cost function

    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__()

        self.__S__ = model
        self.__L__ = trainfunc

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda() 
        outp = self.__S__.forward(data)

        if label == None:
            
            return outp

        else:
            outp = outp.reshape(1, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1

def train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=False):
    # Function to train model

    assert scheduler[1] in ['epoch', 'iteration']
    
    main_model.train()
    device = next(main_model.parameters()).device

    stepsize = train_loader.batch_size

    loss    = 0
    top1    = 0
    counter = 0
    index   = 0

    for data, data_label in train_loader:

        data  = data.to(device)
        label = torch.as_tensor(data_label, dtype=torch.long, device=device)

        ###########################################################
        # Here is your code

        ###########################################################
        
        if verbose:
            print("Epoch {:1.0f}, Batch {:1.0f}, LR {:f} Loss {:f}, Accuracy {:2.3f}%".format(num_epoch, counter, optimizer.param_groups[0]['lr'], loss/counter, top1/counter))

        if scheduler[1] == 'iteration': scheduler[0].step()

    if scheduler[1] == 'epoch': scheduler[0].step()

    return (loss/counter, top1/counter)

def test_network(test_loader, main_model):
    # Function to test model
    
    main_model.eval()
    device = next(main_model.parameters()).device

    loss    = 0
    top1    = 0
    counter = 0

    for data, data_label in test_loader:
        
        data  = data.to(device)
        label = torch.as_tensor(data_label, dtype=torch.long, device=device)

        ###########################################################
        # Here is your code

        ###########################################################
        
    return (loss/counter, top1/counter)

def tar_imp_hists(all_scores, all_labels):
    # Function to divide the scores into target and impostor scores
    
    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):
        
        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])
            
        else:
            imp_scores.append(all_scores[idx])
        
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores