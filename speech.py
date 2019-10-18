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
    speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])
    # speech_recognizer.set_sound_file(loaded_audio['dev-noise-gaussian-5'])
    # enhanced_speech_array = SpeechEnhance(loaded_audio['dev-noise-gaussian-5']).wiener()
    # speech_recognizer.set_audio_array(enhanced_speech_array)
    # speech_recognizer.set_sound_file(speech_enhance.wiener())
    # speech_recognizer.set_sound_file(loaded_audio['clean_sound_file'])

    result = speech_recognizer.deepspeech()
    # result = speech_recognizer.pocketsphinx()

    print('\tActual: ', loaded_audio['transcript_text'])
    print('\tPredicted: ', ' '.join(result))

    ground_truths.append(loaded_audio['transcript_text'])
    hypotheses.append(' '.join(result))

    iterations += 1
word_error_rate = speech_recognizer.word_error_rate(ground_truths, hypotheses)
print(word_error_rate)


quit()

file_path = 'data/LibriSpeech/dev-clean/84/121123/84-121123-0005.flac'

sound_file = SoundFile(file_path)
speech_recognizer = SpeechRecognizer(sound_file)
words = speech_recognizer.pocketsphinx()
# words = speech_recognizer.deepspeech()
print(words)
ground_truth = "D'AVRIGNY UNABLE TO BEAR THE SIGHT OF THIS TOUCHING EMOTION TURNED AWAY AND VILLEFORT WITHOUT SEEKING ANY FURTHER EXPLANATION AND ATTRACTED TOWARDS HIM BY THE IRRESISTIBLE MAGNETISM WHICH DRAWS US TOWARDS THOSE WHO HAVE LOVED THE PEOPLE FOR WHOM WE MOURN EXTENDED HIS HAND TOWARDS THE YOUNG MAN"
word_error_rate = speech_recognizer.word_error_rate(ground_truth, ' '.join(words))
print(word_error_rate)
