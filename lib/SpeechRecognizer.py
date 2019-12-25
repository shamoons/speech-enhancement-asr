from os import path
import pydash
import deepspeech
import re
import editdistance

class SpeechRecognizer:
    def __init__(self, sound_file=None):
        self.SOUND_FILE = sound_file
        self.audio_array = []
        self.samplerate = None
        self.deepspeech_model = None

        self.initialize_deepspeech()

    def set_sound_file(self, sound_file):
        self.SOUND_FILE = sound_file
        self.set_audio_array(self.SOUND_FILE.read(dtype='int16'))
        self.set_samplerate(self.SOUND_FILE.samplerate)

    def set_audio_array(self, audio_array):
        self.audio_array = audio_array

    def set_samplerate(self, samplerate):
        self.samplerate = samplerate

    def initialize_deepspeech(self, model='data/models/deepspeech-0.6.0-models/output_graph.pbmm', alphabet='data/models/deepspeech-0.6.0-models/alphabet.txt', lm='data/models/deepspeech-0.6.0-models/lm.binary', trie='/data/models/deepspeech-0.6.0-models/trie', beam_width=500):
        self.deepspeech_model = deepspeech.Model(model, beam_width)
        self.deepspeech_model.enableDecoderWithLM(lm, trie, 0.75, 1.85)

    def deepspeech(self, audio_array):
        words = self.deepspeech_model.stt(audio_array)
        return words.split(' ')
    # def deepspeech(self):
    #     audio_data = self.audio_array
    #     words = self.deepspeech_model.stt(audio_data)
    #     return words.split(' ')

    def word_distance(self, ground_truth, hypothesis):
        ground_truth_words = ground_truth.split(' ')
        hypothesis_words = hypothesis.split(' ')
        levenshtein_word_distance = editdistance.eval(ground_truth_words, hypothesis_words)

        return levenshtein_word_distance
