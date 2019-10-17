import glob
import os
import soundfile as sf
from soundfile import SoundFile
from lib import NoiseMaker

import numpy as np
import math

clean_path = 'data/LibriSpeech/dev-clean/'

for filepath in glob.iglob(clean_path + '**/*.flac', recursive=True):
    print('Processing: ', filepath)

    target_snrs = np.arange(-5, 11)     # Chosen from https://arxiv.org/pdf/1802.00604.pdf

    sound_file = SoundFile(filepath)
    noise_maker = NoiseMaker(sound_file=sound_file)

    for target_snr in target_snrs:
        gaussian_path = 'dev-noise-gaussian-' + str(target_snr)
        path, file = os.path.split(filepath)
        noisy_path = path.replace('dev-clean', gaussian_path)
        if not os.path.exists(noisy_path):
            os.makedirs(noisy_path)

        noisy_filepath = os.path.join(noisy_path, file)
        noisy_signal = noise_maker.gaussian(target_snr)

        if os.path.exists(noisy_filepath):
            os.remove(noisy_filepath)
        sf.write(noisy_filepath, noisy_signal, sound_file.samplerate)
