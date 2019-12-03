import os
import numpy as np
from lib import NoiseMaker
import pydash
from soundfile import SoundFile

data_path = 'data/LibriSpeech/'
sample_files = []

for root, directories, filenames in os.walk(data_path):

    for filename in filenames:
        filepath = os.path.join(root, filename)
        if ".flac" not in filepath or "-clean" in filepath:
            continue
        sample_files.append(filepath)


selected_files = pydash.sample_size(sample_files, 10)

for selected_file in selected_files:
    (head, tail) = os.path.split(selected_file)
    path_split = selected_file.split('/')
    subset = path_split[2]
    # noise_file = 'data/noise/' + subset.replace('test-noise-', '').split('-')[0] + '.dat'
    subset_split = subset.split('-')
    del subset_split[0:3]
    if len(subset_split) == 2:
        snr = '-' + subset_split[1]
    else:
        snr = subset_split[0]
    snr = int(snr)

    path_split[2] = 'test-clean'

    clean_path = '/'.join(path_split)
    noisy_sound_file = SoundFile(selected_file).read()
    clean_sound_file = SoundFile(clean_path).read()
    noise = noisy_sound_file - clean_sound_file

    clean_norm = np.linalg.norm(clean_sound_file, 2)
    noisy_norm = np.linalg.norm(noise, 2)

    actual_snr = 10 * np.log10(clean_norm / noisy_norm)

    print(snr, actual_snr)
    print('\n')
    continue
