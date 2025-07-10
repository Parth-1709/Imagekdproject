from runpy import run_path
import torch
import os

def get_restormer(device):
    parameters = {
        'inp_channels': 3, 'out_channels': 3, 'dim': 48,
        'num_blocks': [4,6,6,8], 'num_refinement_blocks': 4,
        'heads': [1,2,4,8], 'ffn_expansion_factor': 2.66,
        'bias': False, 'LayerNorm_type': 'WithBias',
        'dual_pixel_task': False
    }
    load_arch = run_path(os.path.join("Restormer", "basicsr", "models", "archs", "restormer_arch.py"))


    model = load_arch['Restormer'](**parameters)
    ckpt = torch.load("Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth")

    model.load_state_dict(ckpt['params'])
    model.to(device).eval()
    return model
