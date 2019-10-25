from scipy.signal import wiener
from soundfile import SoundFile
import soundfile as sf
import os
from lib import SpeechEnhance

file_path = 'data/LibriSpeech/dev-noise-whitenoise-10/84/121123/84-121123-0001.flac'
# sound_file = SoundFile(file_path)
# speech_enhance = SpeechEnhance(sound_file=sound_file)

# print(speech_enhance)

# cleaned_signal = speech_enhance.wiener()

# cleaned_filepath = '84-121123-0001.wiener.flac'

# if os.path.exists(cleaned_filepath):
#     os.remove(cleaned_filepath)
# sf.write(cleaned_filepath, cleaned_signal, sound_file.samplerate)
