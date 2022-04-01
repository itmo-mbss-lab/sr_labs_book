# The script downloads model's weights
# Requirement: wget running on a Linux system 


# Import of modules
import os
import subprocess
from collections import OrderedDict

import torch


def load_model(model, lines, save_path, reload=False):
    # Load model's weights
    
    if not os.path.exists(save_path):
        os.mkdir(save_path, mode=0o777)

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        out = 0

        # Download files
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)
        
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        print('File %s is downloaded.'%outfile)

    # for cpu
    # checkpoint = torch.load(os.path.join(save_path, 'baseline_v2_ap.model'), map_location=torch.device('cpu'))
    # for gpu
    checkpoint = torch.load(os.path.join(save_path, 'baseline_v2_ap.model'))
    
    model_weight = OrderedDict()

    for key in checkpoint.keys():
        
        if '__S__' in key:
            model_weight[key[6:]] = checkpoint[key]
            
    model.load_state_dict(model_weight)
    
    return model