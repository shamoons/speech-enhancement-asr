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

    def initialize_deepspeech(self, model='data/models/deepspeech-0.5.1-models/output_graph.pb', alphabet='data/models/deepspeech-0.5.1-models/alphabet.txt', beam_width=500):
        self.deepspeech_model = deepspeech.Model(model, 26, 9, alphabet, beam_width)

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

        self.pocketsphinx_decoder.end_utt()
        words = pydash.filter_(self.pocketsphinx_decoder.seg(),
                               lambda seg: '<' not in seg.word and '++' not in seg.word and '[' not in seg.word)
        words = pydash.map_(words, lambda seg: re.sub(r'\([^)]*\)', '', seg.word))

        return words

    def deepspeech(self):
        audio_data = self.audio_array
        words = self.deepspeech_model.stt(audio_data, self.samplerate)
        return words.split(' ')

    def word_error_rate(self, ground_truth, hypothesis):
        """Return the Levenshtein edit distance between two strings *a* and *b*."""
        if ground_truth == hypothesis:
            return 0
        if len(ground_truth) < len(hypothesis):
            ground_truth, hypothesis = hypothesis, ground_truth
        if not ground_truth:
            return len(hypothesis)
        previous_row = range(len(hypothesis) + 1)
        for i, column1 in enumerate(ground_truth):
            current_row = [i + 1]
            for j, column2 in enumerate(hypothesis):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (column1 != column2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        levenshtein_distance = previous_row[-1]
        return levenshtein_distance / len(hypothesis)
