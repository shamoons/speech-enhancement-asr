from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from os import path


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
        decoder = Decoder(config)
        decoder.start_utt()

        while self.SOUND_FILE.tell() < len(self.SOUND_FILE):
            print(self.SOUND_FILE.tell(), len(self.SOUND_FILE))
            buf = self.SOUND_FILE.buffer_read(1024, dtype='float32')
            decoder.process_raw(buf, True, False)

        decoder.end_utt()
        print('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
