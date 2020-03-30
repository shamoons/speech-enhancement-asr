import os
import argparse
import soundfile as sf
import numpy as np
from os import path
from wcmatch import glob
from wcmatch import wcmatch

from soundfile import SoundFile
from utilities.noise import add_noise_from_source, add_shift_noise, subtractive_noise

def get_noisy_filepath(clean_filepath, noise, target_snr):
    if '/dev' in clean_filepath:
        source_noise_path = 'dev-noise-' + noise + '-' + str(target_snr)
        path_name, filename = os.path.split(clean_filepath)
        noisy_path = path_name.replace('dev-clean', source_noise_path)
    elif '/train' in clean_filepath:
        source_noise_path = 'train-noise-' + noise + '-' + str(target_snr)
        path_name, filename = os.path.split(clean_filepath)
        noisy_path = path_name.replace('train', source_noise_path)
    elif '/val' in clean_filepath:
        source_noise_path = 'val-noise-' + noise + '-' + str(target_snr)
        path_name, filename = os.path.split(clean_filepath)
        noisy_path = path_name.replace('val', source_noise_path)
    elif '/test' in clean_filepath:
        source_noise_path = 'test-noise-' + noise + '-' + str(target_snr)
        path_name, filename = os.path.split(clean_filepath)
        noisy_path = path_name.replace('test-clean', source_noise_path)


    return noisy_path, filename


def save_noisy_signal(clean_filepath, noise, target_snr, noisy_signal, samplerate):
    noisy_path, filename = get_noisy_filepath(clean_filepath, noise, target_snr)

    if not os.path.exists(noisy_path):
        os.makedirs(noisy_path)

    noisy_filepath = os.path.join(noisy_path, filename)

    if os.path.exists(noisy_filepath):
        os.remove(noisy_filepath)

    sf.write(noisy_filepath, noisy_signal, samplerate)

    return noisy_filepath

def main():
    parser = argparse.ArgumentParser(
        description='Create noisy files.')
    parser.add_argument('--clean_path', default='data/LibriSpeech/test-clean/',
                        help='Path to clean files')

    parser.add_argument('--noise_type', default='additive',
                        help='Noise type')

    parser.add_argument('--ms_to_cut', default='10', help='Milliseconds to cut for subtractive noise')

    parser.add_argument('--save_mask', action='store_true', help='Whether or not to store a mask of missing values')

    args = parser.parse_args()

    clean_path = args.clean_path

    clean_audio_path = os.path.join(clean_path, '**/*.flac')

    for filepath in glob.iglob('**/**/*.flac', root_dir=clean_path, flags=wcmatch.RECURSIVE):
        filepath = os.path.join(clean_path, filepath)
        print('\nProcessing: ', filepath)

        sound_file = SoundFile(filepath)
        sample_rate = sound_file.samplerate
        clean_audio_array = sound_file.read()

        if args.noise_type == 'additive':
            # target_snrs = np.arange(-5, 11)     # Chosen from https://arxiv.org/pdf/1802.00604.pdf
            target_snrs = [0,  5, 10, 15, 20, 25]

            noise_sources = ['white', 'babble', 'f16', 'machinegun', 'm109']

            for target_snr in target_snrs:
                for noise_source in noise_sources:
                    noisy_path, filename = get_noisy_filepath(filepath, noise_source, target_snr)
                    if not path.exists(os.path.join(noisy_path, filename)):
                        noisy_audio_array = add_noise_from_source(clean_audio_array, noise_source, target_snr)
                        noisy_file = save_noisy_signal(filepath, noise_source, target_snr, noisy_audio_array, sample_rate)
                        print('Saved: ', noisy_file)
                noisy_audio_array = add_shift_noise(clean_audio_array, target_snr, 3)
                noisy_file = save_noisy_signal(filepath, 'shift.3', target_snr, noisy_audio_array, sample_rate)
                print('Saved: ', noisy_file)

        elif args.noise_type == 'subtractive':
            ms_to_cut = int(args.ms_to_cut)
            incomplete_audio_array, start_frame, end_frame = subtractive_noise(clean_audio_array, sample_rate, ms_to_cut)
            if len(incomplete_audio_array) > 0:
                noisy_file = save_noisy_signal(filepath, 'subtractive', str(ms_to_cut) + 'ms-1', incomplete_audio_array, sample_rate)

                if args.save_mask:
                    cut_mask = np.zeros(len(clean_audio_array))
                    cut_mask[start_frame:end_frame] = 1
                    mask_filepath = os.path.splitext(noisy_file)[0] +'-mask.npy'
                    print('mask_filepath', mask_filepath)
                    np.savetxt(mask_filepath, cut_mask, fmt='%i')

                print('noisy_file', noisy_file)
            else:
                print('Too short Skipping.')
if __name__ == '__main__':
    main()