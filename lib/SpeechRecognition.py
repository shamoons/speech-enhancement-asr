import deepspeech
import editdistance


class SpeechRecognition:
    def __init__(self):
        self.deepspeech_model = None
        self.initialize_deepspeech()

    def initialize_deepspeech(self, model='data/models/deepspeech-0.6.0-models/output_graph.pbmm', lm='data/models/deepspeech-0.6.0-models/lm.binary', trie='/data/models/deepspeech-0.6.0-models/trie', beam_width=500):
        self.deepspeech_model = deepspeech.Model(model, beam_width)
        self.deepspeech_model.enableDecoderWithLM(lm, trie, 0.75, 1.85)

    def deepspeech(self, audio_array):
        words = self.deepspeech_model.stt(audio_array)
        return words.split(' ')

    def word_distance(self, ground_truth, hypothesis):
        ground_truth_words = ground_truth.split(' ')
        hypothesis_words = hypothesis.split(' ')
        levenshtein_word_distance = editdistance.eval(ground_truth_words, hypothesis_words)

        return levenshtein_word_distance
