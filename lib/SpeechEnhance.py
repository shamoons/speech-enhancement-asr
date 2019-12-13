from scipy.signal import wiener
from .segan_pytorch.segan.models import SEGAN

class SpeechEnhance:
    def __init__(self):
        self.segan = SEGAN()

    def wiener(self, audio_signal):
        return wiener(audio_signal)

    def segan_enhance(self, audio_signal):
        print('hi')
        return
