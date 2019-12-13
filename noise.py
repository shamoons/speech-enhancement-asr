import glob
import os
import soundfile as sf
from soundfile import SoundFile
from lib import NoiseMaker

import numpy as np
import math

clean_path = 'data/LibriSpeech/test-clean/'


def create_gaussian(noise_maker, filepath, target_snr):
    gaussian_path = 'test-noise-gaussian-' + str(target_snr)
    path, file = os.path.split(filepath)
    noisy_path = path.replace('test-clean', gaussian_path)
    if not os.path.exists(noisy_path):
        os.makedirs(noisy_path)

    noisy_filepath = os.path.join(noisy_path, file)
    noisy_signal = noise_maker.gaussian(target_snr)

    if os.path.exists(noisy_filepath):
        os.remove(noisy_filepath)
    sf.write(noisy_filepath, noisy_signal, sound_file.samplerate)


def create_from_source(noise_maker, filepath, target_snr, source):
    source_noise_path = 'test-noise-' + source + '-' + str(target_snr)
    path, file = os.path.split(filepath)
    noisy_path = path.replace('test-clean', source_noise_path)

    if not os.path.exists(noisy_path):
        os.makedirs(noisy_path)

    noisy_filepath = os.path.join(noisy_path, file)

    noisy_signal = noise_maker.source(source, target_snr)

    if os.path.exists(noisy_filepath):
        os.remove(noisy_filepath)

    sf.write(noisy_filepath, noisy_signal, sound_file.samplerate)


for filepath in glob.iglob(clean_path + '**/*.flac', recursive=True):
    print('Processing: ', filepath)

    # target_snrs = np.arange(-5, 11)     # Chosen from https://arxiv.org/pdf/1802.00604.pdf
    target_snrs = [0,  5, 10, 15, 20, 25, 30]

    sound_file = SoundFile(filepath)
    noise_maker = NoiseMaker(sound_file=sound_file)

    for target_snr in target_snrs:
        print('\tTarget SNR: ', target_snr)
        # create_gaussian(noise_maker, filepath, target_snr)
        # create_whitenoise(noise_maker, filepath, target_snr)
        create_from_source(noise_maker, filepath, target_snr, 'white_16k')
        create_from_source(noise_maker, filepath, target_snr, 'babble_16k')
        create_from_source(noise_maker, filepath, target_snr, 'machinegun_16k')
