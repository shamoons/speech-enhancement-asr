from scipy.signal import wiener
# from .segan.model import SEGAN


class SpeechEnhance:

    def wiener(self, audio_signal):
        return wiener(audio_signal)

    def segan(self):
        print('hi')
        return
