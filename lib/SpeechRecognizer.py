from pocketsphinx import get_model_path, Pocketsphinx, Decoder
# from pocketsphinx.pocketsphinx import *
# from sphinxbase.sphinxbase import *
from os import path
from jiwer import wer
import pydash
import deepspeech
import re


class SpeechRecognizer:
    def __init__(self, sound_file=None):
        self.SOUND_FILE = sound_file
        self.audio_array = []
        self.samplerate = None
        self.pocketsphinx_decoder = None
        self.deepspeech_model = None

        self.initialize_pocketsphinx()
        self.initialize_deepspeech()

    def set_sound_file(self, sound_file):
        self.SOUND_FILE = sound_file
        self.set_audio_array(self.SOUND_FILE.read(dtype='int16'))
        self.set_samplerate(self.SOUND_FILE.samplerate)

    def set_audio_array(self, audio_array):
        self.audio_array = audio_array

    def set_samplerate(self, samplerate):
        self.samplerate = samplerate

    def initialize_deepspeech(self):
        return

    def initialize_pocketsphinx(self):
        config = Decoder.default_config()
        config.set_string('-hmm', path.join('data', 'models', 'cmusphinx', 'en-us'))
        config.set_string('-lm', path.join('data', 'models', 'cmusphinx', 'en-us.lm.bin'))
        config.set_string('-dict', path.join('data', 'models', 'cmusphinx', 'cmudict-en-us.dict'))
        config.set_string('-logfn', '/dev/null')
        self.pocketsphinx_decoder = Decoder(config)

    def pocketsphinx(self):
        self.pocketsphinx_decoder.start_utt()

        # How many frames do we want to evaluate? Currently 10ms
        sample_frames = int(self.samplerate / 10)
        current_pos = 0
        while current_pos < len(self.audio_array):
            audio_data = self.audio_array[current_pos:sample_frames + current_pos].tobytes()
            current_pos += sample_frames
            self.pocketsphinx_decoder.process_raw(audio_data, True, False)

        # while self.SOUND_FILE.tell() < len(self.SOUND_FILE):
        #     audio_data = self.SOUND_FILE.read(sample_frames, dtype='int16').tobytes()
        #     self.pocketsphinx_decoder.process_raw(audio_data, True, False)

        self.pocketsphinx_decoder.end_utt()
        words = pydash.filter_(self.pocketsphinx_decoder.seg(),
                               lambda seg: '<' not in seg.word and '++' not in seg.word and '[' not in seg.word)
        words = pydash.map_(words, lambda seg: re.sub(r'\([^)]*\)', '', seg.word))

        return words

    def deepspeech(self, model='data/models/deepspeech-0.5.1-models/output_graph.pb', alphabet='data/models/deepspeech-0.5.1-models/alphabet.txt', beam_width=500):
        ds = deepspeech.Model(aModelPath=model, aAlphabetConfigPath=alphabet,
                              aBeamWidth=beam_width, aNCep=1, aNContext=1, aSampleRate=1)
        # ds = deepspeech.Model(aModelPath=model, aAlphabetConfigPath=alphabet, beam_width)
        audio_data = self.SOUND_FILE.read(dtype='int16').tobytes()
        words = ds.stt(audio_data)
        print(words)

    def word_error_rate(self, ground_truth, hypothesis):
        return wer(ground_truth, hypothesis)
