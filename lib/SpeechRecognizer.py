from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from os import path
from jiwer import wer
import pydash
import deepspeech
import re


class SpeechRecognizer:
    def __init__(self, sound_file):
        self.SOUND_FILE = sound_file

    def pocketsphinx(self,
                     model="/home/shamoon/.local/share/virtualenvs/speech-enhancement-asr-a7TdkoLa/lib/python3.6/site-packages/pocketsphinx/model",
                     ):

        config = Decoder.default_config()
        config.set_string('-hmm', path.join(model, 'en-us'))
        config.set_string('-lm', path.join(model, 'en-us.lm.bin'))
        config.set_string('-dict', path.join(model, 'cmudict-en-us.dict'))
        config.set_string('-logfn', '/dev/null')
        decoder = Decoder(config)
        decoder.start_utt()

        # How many frames do we want to evaluate? Currently 10ms
        sample_frames = int(self.SOUND_FILE.samplerate / 10)

        while self.SOUND_FILE.tell() < len(self.SOUND_FILE):
            audio_data = self.SOUND_FILE.read(sample_frames, dtype='int16').tobytes()
            decoder.process_raw(audio_data, True, False)

        decoder.end_utt()
        words = pydash.filter_(decoder.seg(), lambda seg: '<' not in seg.word)
        words = pydash.map_(words, lambda seg: re.sub(r'\([^)]*\)', '', seg.word))

        return words

    def deepspeech(self, model='/c/Users/Shamoon/Sites/artificial-intelligence/speech-enhancement-asr/data/models/deepspeech-0.5.1-models/lm.binary', alphabet='/c/Users/Shamoon/Sites/artificial-intelligence/speech-enhancement-asr/data/models/deepspeech-0.5.1-models/alphabet.txt', beam_width=500):
        ds = deepspeech.Model(aModelPath=model, aAlphabetConfigPath=alphabet,
                              aBeamWidth=beam_width, aNCep=1, aNContext=1)
        # ds = deepspeech.Model(aModelPath=model, aAlphabetConfigPath=alphabet, beam_width)
        audio_data = self.SOUND_FILE.read(dtype='int16').tobytes()
        words = ds.stt(audio_data)
        print(words)

    def word_error_rate(self, ground_truth, hypothesis):
        return wer(ground_truth, hypothesis)
