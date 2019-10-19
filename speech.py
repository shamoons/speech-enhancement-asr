from lib import SpeechRecognizer, AudioFile, SpeechEnhance
from soundfile import SoundFile

audio_file = AudioFile()
iterations = 0

speech_recognizer = SpeechRecognizer()
ground_truths = []
hypotheses = []

while iterations < 10:
    print(f'Doing Iteration {iterations}')
    loaded_audio = audio_file.load_random()
    print(loaded_audio['dev-noise-gaussian-5'].audio_array)
    quit()
    wiener_enhanced = SpeechEnhance(loaded_audio['dev-noise-gaussian-5'].audio_array).wiener()

    # speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])
    speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # enhanced_speech_array = speech_enhance.wiener(loaded_audio['dev-noise-gaussian-5'])
    speech_recognizer.set_audio_array(wiener_enhanced)

    result = speech_recognizer.deepspeech()
    # result = speech_recognizer.pocketsphinx()

    print('\tActual: ', loaded_audio['transcript_text'])
    print('\tPredicted: ', ' '.join(result))

    ground_truths.append(loaded_audio['transcript_text'])
    hypotheses.append(' '.join(result))

    iterations += 1
word_error_rate = speech_recognizer.word_error_rate(ground_truths, hypotheses)
print(word_error_rate)
