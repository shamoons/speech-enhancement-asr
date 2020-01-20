# pylint: disable=wrong-import-position
import warnings
warnings.simplefilter(action="ignore")
import os

import argparse
import soundfile as sf
import pandas as pd

from pesq import pesq
from pystoi.stoi import stoi
from utilities.files import sample_files, get_transcript
from lib import SpeechRecognition, SpeechEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'


def main():
    speech_recognizer = SpeechRecognition()

    output_df = pd.DataFrame(columns=['audio_file', 'transcript_text', 'predicted_text', 'word_distance',
                                      'word_length', 'pesq', 'stoi'])

    parser = argparse.ArgumentParser(
        description='Calculate WER on speech files by potentially adding noise.')
    parser.add_argument('--enhancement', default='',
                        help='Which enhancement to use')

    parser.add_argument('--noise', default='',
                        help='Noise to test against')

    parser.add_argument('--snr', default='',
                        help='If noise is specified, then the SNR')

    parser.add_argument('--iterations', default=1250,
                        help='Number of iterations')

    parser.add_argument('--save', default='0',
                        help='Save current file')

    args = parser.parse_args()
    noisy_path = 'data/LibriSpeech/'

    if args.noise == '' and args.enhancement == '':
        output_file_name = 'evaluate-clean'
        noisy_path += 'test-clean'
    if args.noise != '':
        output_file_name = 'evaluate-' + args.noise + '-SNR' + args.snr
        noisy_path += 'test-noise-' + args.noise + '-' + args.snr
    if args.enhancement != '':
        output_file_name = output_file_name + '.' + args.enhancement

    output_file_name = output_file_name + '.csv'

    print('Filename: ', output_file_name)

    audio_files = sample_files(args.iterations, path=noisy_path)
    speech_enhance = SpeechEnhance()
    for idx, audio_file in enumerate(audio_files):
        print(f'Doing Iteration {idx}: ', audio_file)

        parts = audio_file.split('/')
        parts[2] = 'test-clean'
        clean_audio_file = '/'.join(parts)

        clean_audio_array, samplerate = sf.read(clean_audio_file)
        noisy_audio_array, samplerate = sf.read(audio_file)

        transcript_text = get_transcript(audio_file)

        if args.enhancement == '':
            audio_array = noisy_audio_array
        elif args.enhancement == 'wiener':
            audio_array = speech_enhance.wiener(noisy_audio_array)
        elif args.enhancement == 'segan':
            audio_array = speech_enhance.segan_enhance(noisy_audio_array)
        elif args.enhancement == 'sevcae':
            audio_array = speech_enhance.sevcae(noisy_audio_array)

        if args.save == '1':
            head_tail = os.path.split(clean_audio_file)
            tail = head_tail[1]
            file_path = tail.split('.')[0]

            sf.write('output/' + file_path + '.clean.wav', clean_audio_array, samplerate)

            if args.noise != '':
                sf.write('output/' + file_path + '.' + args.noise + '.' + args.snr + '.wav', noisy_audio_array, samplerate)

                if args.enhancement != '':
                    sf.write('output/' + file_path + '.' + args.noise + '.' + args.snr + '.' + args.enhancement + '.wav', audio_array, samplerate)

        asr_result = speech_recognizer.deepspeech(audio_array)
        predicted_text = ' '.join(asr_result).upper()
        word_distance = speech_recognizer.word_distance(
            transcript_text, predicted_text)

        calc_pesq = None
        calc_stoi = None

        if args.noise != '':
            calc_pesq = pesq(samplerate, clean_audio_array, audio_array, 'wb')
            calc_stoi = stoi(clean_audio_array, audio_array, samplerate)

        output_df = output_df.append(
            {
                'audio_file': audio_file,
                'transcript_text': transcript_text,
                'predicted_text': predicted_text,
                'word_distance': word_distance,
                'word_length': len(transcript_text.split(' ')),
                'pesq': calc_pesq,
                'stoi': calc_stoi
            }, ignore_index=True)

        if idx % 10 == 0:
            print('Checkpoint saving')
            output_df.to_csv(output_file_name)

    output_df.to_csv(output_file_name)

    print(output_df)
    print(output_file_name)

    # print('T', transcript_text)
    # print('P', predicted_text)
    # print('WD', word_distance)
    # print('\n')


if __name__ == '__main__':
    main()
