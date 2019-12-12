from lib import SpeechRecognizer, AudioFile, SpeechEnhance
from pesq import pesq
from pystoi.stoi import stoi

import pandas as pd
import os
import argparse

audio_file = AudioFile()
iterations = 0

speech_recognizer = SpeechRecognizer()
speech_enhance = SpeechEnhance()
ground_truths = []
hypotheses = []

output_df = pd.DataFrame(columns=['book_id', 'chapter_id', 'transcript_id',
                                  'transcript_text', 'predicted_text', 'word_distance',
                                  'word_length', 'pesq', 'stoi'])

parser = argparse.ArgumentParser(
    description='Calculate WER on speech files from a subset.')
parser.add_argument('--subset', default='test-clean',
                    help='subset to choose from')

parser.add_argument('--enhancement', default='',
                    help='Which enhancement to use')
args = parser.parse_args()

output_file = args.subset
if args.enhancement != '':
    output_file += '.' + args.enhancement
output_file += '.csv'
print('Filename', output_file)

while iterations < 1250:
    loaded_audio = audio_file.load_random(subset=args.subset)
    print(f'Doing Iteration {iterations}')

    speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])

    clean_result = speech_recognizer.deepspeech()
    sound_file_subset = loaded_audio['subset']
    predicted_text = ' '.join(clean_result).upper()

    word_distance = speech_recognizer.word_distance(
        loaded_audio['transcript_text'], predicted_text)

    calc_pesq = None
    calc_stoi = None
    if args.subset != 'test-clean':
        noisy_audio_array = speech_recognizer.audio_array
        clean_audio = audio_file.load(
            loaded_audio['book_id'], loaded_audio['chapter_id'], loaded_audio['transcript_id'], 'test-clean')
        clean_audio_array = clean_audio['clean_sound_file'].read()
        

        comparison_audio_array = clean_audio_array
        if args.enhancement == 'wiener':
            comparison_audio_array = speech_enhance.wiener(noisy_audio_array)


        calc_pesq = pesq( speech_recognizer.samplerate, clean_audio_array, noisy_audio_array, 'wb')
        calc_stoi = stoi(clean_audio_array, noisy_audio_array, speech_recognizer.samplerate)

    output_df = output_df.append(
        {
            'book_id': loaded_audio['book_id'],
            'chapter_id': loaded_audio['chapter_id'],
            'transcript_id': loaded_audio['transcript_id'],
            'transcript_text': loaded_audio['transcript_text'],
            'predicted_text': predicted_text,
            'word_distance': word_distance,
            'word_length': len(loaded_audio['transcript_text'].split(' ')),
            'pesq': calc_pesq,
            'stoi': calc_stoi
        }, ignore_index=True)

    if iterations % 10 == 0:
        print('Checkpoint saving')
        output_df.to_csv(output_file)

    iterations += 1

word_distance_sum = output_df['word_distance'].sum()
word_length_sum = output_df['word_length'].sum()
print('word_distance_sum', word_distance_sum)
print('word_length_sum', word_length_sum)

output_df.to_csv(output_file)

print(output_df)
print(output_file)
