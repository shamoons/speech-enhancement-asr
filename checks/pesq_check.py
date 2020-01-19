import soundfile as sf
from pesq import pesq


def main():


    print('Testing Max Pesq')
    clean_audio_array, samplerate = sf.read('data/LibriSpeech/test-clean/121/121726/121-121726-0003.flac')
    calc_pesq = pesq(samplerate, clean_audio_array, clean_audio_array, 'wb')
    print('PESQ: ', calc_pesq)

    print('Testing Pesq (Babble-5)')
    noisy_audio_array, samplerate = sf.read('data/LibriSpeech/test-noise-babble-5/121/121726/121-121726-0003.flac')
    calc_pesq = pesq(samplerate, clean_audio_array, noisy_audio_array, 'wb')
    print('PESQ: ', calc_pesq)

if __name__ == '__main__':
    main()