import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

audio_signal, samplerate = sf.read("data/LibriSpeech/dev-clean/84/121550/84-121550-0035.flac")

print('Sample Rate: ', samplerate)

print('Signal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] /
                                float(samplerate), 2), 'seconds')
print(audio_signal)

audio_signal = audio_signal / np.power(2, 15)
length_signal = len(audio_signal)
half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)

signal_frequency = np.fft.fft(audio_signal)

signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
signal_frequency **= 2

len_fts = len(signal_frequency)

if length_signal % 2:
    signal_frequency[1:len_fts] *= 2
else:
    signal_frequency[1:len_fts-1] *= 2

signal_power = 10 * np.log10(signal_frequency)

x_axis = np.arange(0, half_length, 1) * (samplerate / length_signal) / 1000.0

plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
# plt.show()
plt.savefig('temp.png')
