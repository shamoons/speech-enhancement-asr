from scipy.signal import wiener
import json
import torch
import soundfile as sf
from .segan_pytorch.segan.models import SEGAN
from .segan_pytorch.segan.datasets import normalize_wave_minmax
from .segan_pytorch.segan.datasets import pre_emphasize

class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

class SpeechEnhance:
    def __init__(self):
        with open("data/models/segan_v1.1/train.opts", 'r') as cfg_f:
            self.segan_args = ArgParser(json.load(cfg_f))
        # {
        #     "g_pretrained_ckpt": "data/models/segan_v1.1/segan+_generator.ckpt",
        #     "seed": 111,
        #     "synthesis_path": "segan_enhanced"

        # }


        self.segan = SEGAN(self.segan_args)
        self.segan.G.load_pretrained("data/models/segan_v1.1/segan+_generator.ckpt", True)
        self.segan.G.eval()

    def wiener(self, audio_signal):
        return wiener(audio_signal)

    def segan_enhance(self, audio_signal):
        wav = normalize_wave_minmax(audio_signal)
        wav = pre_emphasize(wav)
        pwav = torch.FloatTensor(wav).view(1,1,-1)

        g_wav, g_c = self.segan.generate(pwav)
        sf.write('file.wav', g_wav, 16000)

        return g_wav
