import numpy as np


class NoiseMaker:
    def __init__(self, sound_file):
        self.SOUND_FILE = sound_file
        self.audio_signal = sound_file.read()

    def gaussian(self, target_snr):
        # https: // stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        audio_power = self.audio_signal ** 2

        sig_avg = np.mean(audio_power)
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - target_snr
        noise_avg = 10 ** (noise_avg_db / 10)

        noise = np.random.normal(0, np.sqrt(noise_avg), self.audio_signal.shape[0])

        noisy_signal = self.audio_signal + noise

        return noisy_signal
