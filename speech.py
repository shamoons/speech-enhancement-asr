from os import path
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from soundfile import SoundFile

MODELDIR = "/home/shamoon/.local/share/virtualenvs/speech-enhancement-asr-a7TdkoLa/lib/python3.6/site-packages/pocketsphinx/model"
DATADIR = "/home/shamoon/.local/share/virtualenvs/speech-enhancement-asr-a7TdkoLa/lib/python3.6/site-packages/pocketsphinx/data"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'cmudict-en-us.dict'))
decoder = Decoder(config)
decoder.start_utt()


# stream = open('84-121123-0000.wav', 'rb')
file_path = 'data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac'
sound_file = SoundFile(file_path)
# filename, file_extension = path.splitext(file_path)
# flac_tmp_audio_data = AudioSegment.from_file(file_path, file_extension[1:])

# print(flac_tmp_audio_data)
# quit()

# stream = open('data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac', 'rb')
# stream = open(path.join(DATADIR, 'goforward.raw'), 'rb')
while sound_file.tell() < len(sound_file):
    print(sound_file.tell(), len(sound_file))

    # buf = stream.read(1024)
    # print(buf)
    # quit()
    buf = sound_file.buffer_read(1024, dtype='float32')
    # current_pos += 1024
    # if buf:
    decoder.process_raw(buf, True, False)
    print('here i am!')
    print(buf)
    # break

    # else:
    #     break
decoder.end_utt()
print('Best hypothesis segments: ', [seg.word for seg in decoder.seg()])
