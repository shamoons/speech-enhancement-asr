from lib import SpeechRecognizer, AudioFile, SpeechEnhance
from soundfile import SoundFile
import pandas as pd
import argparse

audio_file = AudioFile()
iterations = 0

speech_recognizer = SpeechRecognizer()
ground_truths = []
hypotheses = []

output_df = pd.DataFrame(columns=['book_id', 'chapter_id', 'transcript_id',
                                  'transcript_text', 'clean_predicted_text', 'clean_word_distance', 'word_length'])

parser = argparse.ArgumentParser(description='Calculate WER on speech files from a subset.')
parser.add_argument('--subset', default='test-clean', help='subset to choose from')
args = parser.parse_args()

output_file = args.subset + '.csv'
print('Filename', output_file)

while iterations < 50:
    loaded_audio = audio_file.load_random(subset=args.subset)
    print(f'Doing Iteration {iterations}')

    # wiener_enhanced = SpeechEnhance(loaded_audio['dev-noise-gaussian-5'].audio_array).wiener()

    speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])
    # speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # enhanced_speech_array = speech_enhance.wiener(loaded_audio['dev-noise-gaussian-5'])
    # speech_recognizer.set_audio_array(wiener_enhanced)

    # result = speech_recognizer.pocketsphinx()
    clean_result = speech_recognizer.deepspeech()
    clean_predicted_text = ' '.join(clean_result).upper()

    clean_word_distance = speech_recognizer.word_distance(loaded_audio['transcript_text'], clean_predicted_text)

    # speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # result = speech_recognizer.deepspeech()
    # predicted_text = ' '.join(result).upper()

    # word_error_rate = speech_recognizer.word_error_rate(loaded_audio['transcript_text'], predicted_text)

    # print('\tActual: ', loaded_audio['transcript_text'])
    # print('\tPredicted: ', predicted_text)
    # print('\tWER: ', clean_word_distance)
    # print('\t5 WER: ', word_error_rate)

    output_df = output_df.append(
        {
            'book_id': loaded_audio['book_id'],
            'chapter_id': loaded_audio['chapter_id'],
            'transcript_id': loaded_audio['transcript_id'],
            'transcript_text': loaded_audio['transcript_text'],
            'clean_predicted_text': clean_predicted_text,
            'clean_word_distance': clean_word_distance,
            'word_length': len(loaded_audio['transcript_text'].split(' '))
        }, ignore_index=True)

    if iterations % 10 == 0:
        print('Checkpoint saving')
        output_df.to_csv(output_file)

    iterations += 1

print(output_file)
print(output_df)
output_df.to_csv(output_file)
