# The script is borrowed from the following repository: https://github.com/clovaai/voxceleb_trainer
# The script allows to download data from dataset


# Import of modules
import os
import glob

import numpy
import random
import soundfile
from scipy import signal

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Load wav file
    
    max_audio = max_frames*160 + 240

    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage  = max_audio - audiosize + 1 
        audio     = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize - max_audio))])
    
    feats = []
    
    if evalmode and max_frames == 0:
        feats.append(audio)
    
    else:
        
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(numpy.float)

    return feat

class test_dataset_loader(Dataset):
    # Test dataset loader
    
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        
        audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        
        return len(self.test_list)

class AugmentWAV(object):
    # Augmentation to wav files

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames*160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr   = {'noise':[0, 15], 'speech':[13, 20], 'music':[5, 15]}
        self.numnoise   = {'noise':[1, 1],  'speech':[3, 7],   'music':[1, 1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'));

        for file in augment_files:
            
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path, '*/*/*.wav'));

    def additive_noise(self, noisecat, audio):
        # Augmentation by additive noise

        clean_db  = 10*numpy.log10(numpy.mean(audio**2) + 1e-4)

        numnoise  = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr  = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db   = 10*numpy.log10(numpy.mean(noiseaudio[0]**2) + 1e-4) 

            noises.append(numpy.sqrt(10**((clean_db - noise_db - noise_snr)/10))*noiseaudio)

        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        # Augmentation by reverberation

        rir_file = random.choice(self.rir_files)
        
        rir, fs  = soundfile.read(rir_file)
        rir      = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir      = rir/numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]