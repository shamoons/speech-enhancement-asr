import json
from scipy.signal import wiener
import tensorflow as tf
import torch

import numpy as np
from scipy.signal import wiener
from .segan_pytorch.segan.models import SEGAN
from .segan_pytorch.segan.datasets import normalize_wave_minmax
from .segan_pytorch.segan.datasets import pre_emphasize
from .Se_Vcae import SeVcae


class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


class SpeechEnhance:
    def __init__(self):
        with open("data/models/segan_v1.1/train.opts", 'r') as cfg_f:
            self.segan_args = ArgParser(json.load(cfg_f))

        self.segan = SEGAN(self.segan_args)
        self.segan.G.load_pretrained(
            "data/models/segan_v1.1/segan+_generator.ckpt", True)
        self.segan.G.eval()

        self.se_vcae = SeVcae('data/models/se_vcae/DN_VCAE_330lf_w600.ckpt')

    def convert_to_int(self, audio_signal):
        enhanced_signal = (audio_signal * 32767).astype(np.int16)
        return enhanced_signal
     
    def convert_to_float(self, audio_signal):
        float_audio_signal = (audio_signal / 32767).astype(np.float64)
        return float_audio_signal 

    def wiener(self, audio_signal):
        enhanced_signal = wiener(audio_signal)

        return enhanced_signal

    def segan_enhance(self, audio_signal):
        float_audio_signal = self.convert_to_int(audio_signal)
        wav = normalize_wave_minmax(float_audio_signal)
        wav = pre_emphasize(wav)
        pwav = torch.FloatTensor(wav).view(1, 1, -1)

        g_wav, g_c = self.segan.generate(pwav)

        return g_wav
    
    def sevcae(self, audio_signal):

        enhanced_signal = self.se_vcae.enhance(audio_signal)
        return enhanced_signal
