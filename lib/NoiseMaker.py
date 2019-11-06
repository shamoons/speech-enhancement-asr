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

    def source(self, source, target_snr):
        source_noise = np.fromfile('data/noise/' + source + '.dat', sep='\n')
        source_noise_start = np.random.randint(0, len(source_noise) - len(self.audio_signal))
        source_noise = source_noise[source_noise_start:source_noise_start + len(self.audio_signal)]

        source_noise_norm = np.linalg.norm(source_noise, 2)
        source_noise_norm = np.sqrt(np.sum(source_noise ** 2))
        signal_norm = np.linalg.norm(self.audio_signal, 2)
        desired_noise_norm = signal_norm / 10 ** (target_snr / 20)
        ratio = desired_noise_norm / source_noise_norm

        noisy_signal = self.audio_signal + ratio * source_noise

        return noisy_signal

    def white(self, target_snr):
        white_noise = np.fromfile('data/noise/white.dat', sep='\n')
        white_noise_start = np.random.randint(0, len(white_noise) - len(self.audio_signal))
        white_noise = white_noise[white_noise_start:white_noise_start + len(self.audio_signal)]

        white_noise_norm = np.linalg.norm(white_noise, 2)
        white_noise_norm = np.sqrt(np.sum(white_noise ** 2))
        signal_norm = np.linalg.norm(self.audio_signal, 2)
        desired_noise_norm = signal_norm / 10 ** (target_snr / 20)
        ratio = desired_noise_norm / white_noise_norm

        noisy_signal = self.audio_signal + ratio * white_noise

        return noisy_signal
