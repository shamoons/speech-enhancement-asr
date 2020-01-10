import glob
import os
from os import path
import soundfile as sf
from soundfile import SoundFile
from utilities.noise import add_noise_from_source, add_shift_noise

# def create_gaussian(noise_maker, filepath, target_snr):
#     gaussian_path = 'test-noise-gaussian-' + str(target_snr)
#     path, file = os.path.split(filepath)
#     noisy_path = path.replace('test-clean', gaussian_path)
#     if not os.path.exists(noisy_path):
#         os.makedirs(noisy_path)

#     noisy_filepath = os.path.join(noisy_path, file)
#     noisy_signal = noise_maker.gaussian(target_snr)

#     if os.path.exists(noisy_filepath):
#         os.remove(noisy_filepath)
#     sf.write(noisy_filepath, noisy_signal, sound_file.samplerate)


# def create_from_source(noise_maker, filepath, target_snr, source):
#     source_noise_path = 'test-noise-' + source + '-' + str(target_snr)
#     path, file = os.path.split(filepath)
#     noisy_path = path.replace('test-clean', source_noise_path)

#     if not os.path.exists(noisy_path):
#         os.makedirs(noisy_path)

#     noisy_filepath = os.path.join(noisy_path, file)

#     noisy_signal = noise_maker.source(source, target_snr)

#     if os.path.exists(noisy_filepath):
#         os.remove(noisy_filepath)

#     sf.write(noisy_filepath, noisy_signal, sound_file.samplerate)

def get_noisy_filepath(clean_filepath, noise, target_snr):
    source_noise_path = 'test-noise-' + noise + '-' + str(target_snr)
    path_name, filename = os.path.split(clean_filepath)
    noisy_path = path_name.replace('test-clean', source_noise_path)

    return noisy_path, filename


def save_noisy_signal(clean_filepath, noise, target_snr, noisy_signal, samplerate):
    noisy_path, filename = get_noisy_filepath(clean_filepath, noise, target_snr)

    # source_noise_path = 'test-noise-' + noise + '-' + str(target_snr)
    # path_name, file = os.path.split(clean_filepath)
    # noisy_path = path_name.replace('test-clean', source_noise_path)

    if not os.path.exists(noisy_path):
        os.makedirs(noisy_path)

    noisy_filepath = os.path.join(noisy_path, filename)

    if os.path.exists(noisy_filepath):
        os.remove(noisy_filepath)

    sf.write(noisy_filepath, noisy_signal, samplerate)

    return noisy_filepath



def main():
    clean_path = 'data/LibriSpeech/test-clean/'

    for filepath in glob.iglob(clean_path + '**/*.flac', recursive=True):
        print('Processing: ', filepath)

        # target_snrs = np.arange(-5, 11)     # Chosen from https://arxiv.org/pdf/1802.00604.pdf
        target_snrs = [0,  5, 10, 15, 20, 25]

        sound_file = SoundFile(filepath)
        sample_rate = sound_file.samplerate
        clean_audio_array = sound_file.read()

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

if __name__ == '__main__':
    main()