import os
from sklearn.preprocessing import minmax_scale
import pydash
import numpy as np


def add_noise(audio_array, source_noise, target_snr):
    source_noise_norm = np.linalg.norm(source_noise, 2)
    signal_norm = np.linalg.norm(audio_array, 2)
    desired_noise_norm = signal_norm / 10 ** (target_snr / 20)
    ratio = desired_noise_norm / source_noise_norm

    while len(source_noise) < len(audio_array):
        source_noise = np.append(source_noise, 0)
    noisy_signal = audio_array + ratio * source_noise

    # noisy_signal = noisy_signal.astype(np.int16)

    # noise = noisy_signal - audio_array
    # clean_norm = np.linalg.norm(audio_array, 2)
    # noisy_norm = np.linalg.norm(noise, 2)

    # actual_snr = 20 * np.log10(clean_norm / noisy_norm)
    # print('Target SNR: ', target_snr, 'Actual SNR: ', actual_snr)

    return noisy_signal

def read_noise_file(audio_array_length, source):
    source_noise = np.fromfile('data/noise/' + source + '_16k.dat', sep='\n')
    source_noise_start = np.random.randint(
        0, len(source_noise) - audio_array_length)
    source_noise = source_noise[source_noise_start:
                                source_noise_start + audio_array_length]
    source_noise = minmax_scale(source_noise, feature_range=(-100, 100))

    return source_noise


def add_noise_from_source(audio_array, source, target_snr):
    target_snr = float(target_snr)
    source_noise = read_noise_file(len(audio_array), source)

    return add_noise(audio_array, source_noise, target_snr)

def add_shift_noise(audio_array, target_snr, num_slices=3, path='data/noise'):
    target_snr = float(target_snr)

    source_noises = os.listdir(path)
    source_noises = pydash.filter_(source_noises, lambda source_noise: source_noise.__contains__('.dat'))
    # source_noises.append('')
    selected_noises = pydash.sample_size(source_noises, num_slices)

    source_noise = []
    slice_length = int(len(audio_array) / num_slices)
    for selected_noise in selected_noises:
        selected_noise = selected_noise.replace('_16k.dat', '')
        if selected_noise == '':
            # We are not adding noise, so we can add zeroes
            slice_source_noise = np.zeros(slice_length)
        else:
            slice_source_noise = read_noise_file(slice_length, selected_noise)

            source_noise = np.concatenate([source_noise, slice_source_noise])

    return add_noise(audio_array, source_noise, target_snr)
