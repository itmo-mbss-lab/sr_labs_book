# Exercises in order to perform laboratory work


# Import of modules
import os
import random

import numpy

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import Dataset

from common import loadWAV, AugmentWAV
from ResNetBlocks import *
from preproc import PreEmphasis


class train_dataset_loader(Dataset):
    # Train dataset loader
    
    def __init__(self, train_list, max_frames, train_path, augment=False, musan_path=None, rir_path=None):

        self.max_frames  = max_frames
        self.augment     = augment

        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in train_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
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

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in test_list]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
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

class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        
        return torch.max(self.linear1(x), self.linear2(x))

class ResNet(nn.Module):
    # ResNet model for speaker recognition

    def __init__(self, block, layers, activation, num_filters, nOut, encoder_type='SP', n_mels=64, log_input=True, **kwargs):
        
        super(ResNet, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.inplanes     = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.torchfb        = torch.nn.Sequential(PreEmphasis(), 
                                                  torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                                       n_fft=512, 
                                                                                       win_length=400, 
                                                                                       hop_length=160, 
                                                                                       window_fn=torch.hamming_window, 
                                                                                       n_mels=n_mels))
        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1  = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(num_filters[0])
        self.relu   = activation(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2, activation=activation)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(nn.Conv1d(num_filters[3]*outmap_size, 128, kernel_size=1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(128), 
                                       nn.Conv1d(128, num_filters[3]*outmap_size, kernel_size=1), 
                                       nn.Softmax(dim=2))
        
        if self.encoder_type == "SP":
            out_dim = num_filters[3]*outmap_size*2
        
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3]*outmap_size*2
        
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Sequential(MaxoutLinear(out_dim, nOut), nn.BatchNorm1d(nOut, affine=False))

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        
        return out

    def forward(self, x):

        with torch.no_grad():
            
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                
                if self.log_input: x = x.log()
                
                x = self.instancenorm(x).unsqueeze(1)

        ###########################################################
        # Here is your code

        ###########################################################

        return x

class MainModel(nn.Module):

    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__();

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
    
    stepsize = train_loader.batch_size

    loss    = 0
    top1    = 0
    counter = 0
    index   = 0

    for data, data_label in train_loader:

        data = data.transpose(1, 0)
        
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

    loss    = 0
    top1    = 0
    counter = 0

    for data, data_label in test_loader:
        
        data = data.transpose(1, 0)
        
        ###########################################################
        # Here is your code
        
        ###########################################################

    return (loss/counter, top1/counter)