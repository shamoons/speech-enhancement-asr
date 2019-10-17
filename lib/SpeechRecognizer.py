from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from os import path
from jiwer import wer

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
            audio_data = self.SOUND_FILE.read(sample_frames, dtype='int16')
            decoder.process_raw(audio_data.tobytes(), True, False)

        decoder.end_utt()
        words = []
        for seg in decoder.seg():
            word = re.sub(r'\([^)]*\)', '', seg.word)

            words.append(word)
        return words

    def word_error_rate(self, ground_truth, hypothesis):
        return wer(ground_truth, hypothesis)
