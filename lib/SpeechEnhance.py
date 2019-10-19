from scipy.signal import wiener


class SpeechEnhance:
    def __init__(self, audio_array):
        self.audio_array = audio_array

    def wiener(self):
        print(self.audio_array)
        return wiener(self.audio_array)
