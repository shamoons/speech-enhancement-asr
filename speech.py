from lib import SpeechRecognizer
from soundfile import SoundFile

file_path = 'data/LibriSpeech/dev-clean/84/121123/84-121123-0005.flac'

sound_file = SoundFile(file_path)
speech_recognizer = SpeechRecognizer(sound_file)
# words = speech_recognizer.pocketsphinx()
words = speech_recognizer.deepspeech()
# print('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
print(words)
ground_truth = "D'AVRIGNY UNABLE TO BEAR THE SIGHT OF THIS TOUCHING EMOTION TURNED AWAY AND VILLEFORT WITHOUT SEEKING ANY FURTHER EXPLANATION AND ATTRACTED TOWARDS HIM BY THE IRRESISTIBLE MAGNETISM WHICH DRAWS US TOWARDS THOSE WHO HAVE LOVED THE PEOPLE FOR WHOM WE MOURN EXTENDED HIS HAND TOWARDS THE YOUNG MAN"
word_error_rate = speech_recognizer.word_error_rate(ground_truth, ' '.join(words))
print(word_error_rate)
