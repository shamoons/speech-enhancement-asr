import glob
import os
import soundfile as sf
import numpy as np
import math

clean_path = 'data/LibriSpeech/dev-clean/'

for filepath in glob.iglob(clean_path + '**/*.flac', recursive=True):
    target_snrs = [-10, 0, 10, 20, 30]

    for target_snr in target_snrs:
        gaussian_path = 'dev-noise-gaussian-' + str(target_snr)
        print('Processing: ', filepath)
        path, file = os.path.split(filepath)
        noisy_path = path.replace('dev-clean', gaussian_path)
        if not os.path.exists(noisy_path):
            os.makedirs(noisy_path)

        noisy_filepath = os.path.join(noisy_path, file)
        audio_signal, samplerate = sf.read(filepath)

        audio_power = audio_signal ** 2

        sig_avg = np.mean(audio_power)
        sig_avg_db = 10 * np.log10(sig_avg)
        noise_avg_db = sig_avg_db - target_snr
        noise_avg = 10 ** (noise_avg_db / 10)

        noise = np.random.normal(0, np.sqrt(noise_avg), audio_signal.shape[0])

        noisy_signal = audio_signal + noise

        if os.path.exists(noisy_filepath):
            os.remove(noisy_filepath)
        sf.write(noisy_filepath, noisy_signal, samplerate)
