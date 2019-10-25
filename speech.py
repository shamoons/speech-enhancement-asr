from lib import SpeechRecognizer, AudioFile, SpeechEnhance
from soundfile import SoundFile
import pandas as pd

audio_file = AudioFile()
iterations = 0

speech_recognizer = SpeechRecognizer()
ground_truths = []
hypotheses = []

output_df = pd.DataFrame(columns=['book_id', 'chapter_id', 'transcript_id',
                                  'transcript_text', 'clean_predicted_text', 'clean_wer'])

while iterations < 250:
    print(f'Doing Iteration {iterations}')
    loaded_audio = audio_file.load_random()

    # wiener_enhanced = SpeechEnhance(loaded_audio['dev-noise-gaussian-5'].audio_array).wiener()

    speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])
    # speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # enhanced_speech_array = speech_enhance.wiener(loaded_audio['dev-noise-gaussian-5'])
    # speech_recognizer.set_audio_array(wiener_enhanced)

    # result = speech_recognizer.pocketsphinx()
    clean_result = speech_recognizer.deepspeech()
    clean_predicted_text = ' '.join(clean_result).upper()

    clean_word_error_rate = speech_recognizer.word_error_rate(loaded_audio['transcript_text'], clean_predicted_text)

    # speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # result = speech_recognizer.deepspeech()
    # predicted_text = ' '.join(result).upper()

    # word_error_rate = speech_recognizer.word_error_rate(loaded_audio['transcript_text'], predicted_text)


    # print('\tActual: ', loaded_audio['transcript_text'])
    # print('\tPredicted: ', predicted_text)
    # print('\tWER: ', clean_word_error_rate)
    # print('\t5 WER: ', word_error_rate)

    output_df = output_df.append(
        {
            'book_id': loaded_audio['book_id'],
            'chapter_id': loaded_audio['chapter_id'],
            'transcript_id': loaded_audio['transcript_id'],
            'transcript_text': loaded_audio['transcript_text'],
            'clean_predicted_text': clean_predicted_text,
            'clean_wer': clean_word_error_rate,
#            '5-snr-wer': word_error_rate
        }, ignore_index=True)

    if iterations % 10 == 0:
        print('Checkpoint saving')
        output_df.to_csv('output_df.csv')

    iterations += 1
# word_error_rate = speech_recognizer.word_error_rate(ground_truths, hypotheses)
# print(word_error_rate)
print(output_df)
output_df.to_csv('output_df.csv')
