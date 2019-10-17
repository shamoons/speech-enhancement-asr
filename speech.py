from os import environ, path

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = "/home/shamoon/.local/share/virtualenvs/speech-enhancement-asr-a7TdkoLa/lib/python3.6/site-packages/pocketsphinx/model"
DATADIR = "/home/shamoon/.local/share/virtualenvs/speech-enhancement-asr-a7TdkoLa/lib/python3.6/site-packages/pocketsphinx/data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'cmudict-en-us.dict'))
decoder = Decoder(config)

# Decode streaming data.
decoder = Decoder(config)
decoder.start_utt()
stream = open('84-121123-0000.wav', 'rb')
# stream = open('data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac', 'rb')
# stream = open(path.join(DATADIR, 'goforward.raw'), 'rb')
while True:
    buf = stream.read(1024)
    if buf:
        decoder.process_raw(buf, True, False)
    else:
        break
decoder.end_utt()
print('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
