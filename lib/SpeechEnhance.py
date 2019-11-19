from scipy.signal import wiener
# from .segan.model import SEGAN


class SpeechEnhance:
    def __init__(self, sound_file):
        self.SOUND_FILE = sound_file
        self.audio_signal = sound_file.read()

    def wiener(self):
        return wiener(self.audio_signal)

    def segan(self):
        print('hi')
        return
