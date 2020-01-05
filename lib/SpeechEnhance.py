import json
from scipy.signal import wiener
import torch
import tensorflow as tf
import numpy as np
from scipy.signal import wiener
from .segan_pytorch.segan.models import SEGAN
from .segan_pytorch.segan.datasets import normalize_wave_minmax
from .segan_pytorch.segan.datasets import pre_emphasize
# from .DeepXiNet import DeepXiNet

class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

class SpeechEnhance:
    def __init__(self):
        with open("data/models/segan_v1.1/train.opts", 'r') as cfg_f:
            self.segan_args = ArgParser(json.load(cfg_f))

        self.segan = SEGAN(self.segan_args)
        self.segan.G.load_pretrained("data/models/segan_v1.1/segan+_generator.ckpt", True)
        self.segan.G.eval()

        # self.deepxi = DeepXiNet()
    def convert_to_int(self, audio_signal):
        enhanced_signal = (audio_signal * 32767).astype(np.int16)
        return enhanced_signal

    def wiener(self, audio_signal):
        enhanced_signal = wiener(audio_signal)
        print('audio_signal')
        print(audio_signal)
        print(np.min(audio_signal),np.max(audio_signal))

        print('enhanced_signal')
        # enhanced_signal = (enhanced_signal).astype(np.int16)
        print(enhanced_signal)
        print(np.min(enhanced_signal),np.max(enhanced_signal))
        return enhanced_signal
        # return self.convert_to_int(wiener(audio_signal))

    def segan_enhance(self, audio_signal):
        wav = normalize_wave_minmax(audio_signal)
        wav = pre_emphasize(wav)
        pwav = torch.FloatTensor(wav).view(1, 1, -1)

        g_wav, g_c = self.segan.generate(pwav)

        return self.convert_to_int(g_wav)
    
    # def deepxi_enhance(self, audio_signal):
    #     with tf.Session() as sess:
    #         self.deepxi.saver.restore(sess, 'lib/deepxi/model/3e/epoch-173')
    #         return
