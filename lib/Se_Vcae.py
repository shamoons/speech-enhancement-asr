import tensorflow as tf
import numpy as np
from .sevcae.SE_VCAE import encoder
from .sevcae.SE_VCAE import decoder
from .sevcae.SE_VCAE import de_emph

class SeVcae:
    def __init__(self, model_name):
        self.model_name = model_name
        self.opts = {
            "z_dim": 330,
            "peek": 200,
            "cs": 600,
            "overlap_p": 0.5,
            "hann_window": np.hanning(600)
        }

        self.opts["overlap_c"] = int(self.opts["cs"]  * self.opts["overlap_p"] )

    def pre_emph(self, x, coeff=0.95):
        x0 = np.reshape(x[0], [1,])
        diff = x[1:] - coeff * x[:-1]
        concat = np.concatenate([x0, diff], axis=0)
        return concat

    def get_chunk_with_margin(self, x, i, cs, peek):
        min_i = max([i-peek, 0])
        max_i = min([len(x), i + cs + peek])
        chunk = x[min_i: max_i]

        # Check if the we need to pad with 0's
        if i - peek < 0:
            diff = np.abs(i - peek)
            chunk = np.concatenate([np.zeros(diff), chunk], axis=0)
        if i + cs + peek >= len(x):
            diff = np.abs(len(x) - i - peek - cs)
            chunk = np.concatenate([chunk, np.zeros(diff)], axis=0)

        return chunk

    def get_chunk(self, audio_signal):
        # Split sound files into chunks.
        X = []
        pre_emph_signal = self.pre_emph(audio_signal)

        # Iterate through the audio file's samples constructing blocks.
        for i in range(0, len(pre_emph_signal), self.opts["cs"] - self.opts["overlap_c"]):
            raw_chunk = pre_emph_signal[i:i + self.opts["cs"]]
            # Check if we need to pad raw_chunk on the right, i.e. are we
            # at the end of the file.
            if len(raw_chunk) < self.opts["cs"]:
                diff = self.opts["cs"] - len(raw_chunk)
                raw_chunk = np.concatenate([raw_chunk, np.zeros(diff)], axis=0)

            padded_chunk = self.get_chunk_with_margin(pre_emph_signal, i, self.opts["cs"], self.opts["peek"])
            
            X.append(padded_chunk)
        
        return X

    def enhance(self, audio_signal):
        chunk_signal = self.get_chunk(audio_signal)

        X_n = tf.placeholder(tf.float32, shape=[None, 1000])
        X = tf.placeholder(tf.float32, shape=[None, 600])

        Z_mu = encoder(X_n, self.opts["z_dim"])
        Z = Z_mu

        X_hat = decoder(Z)

        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.model_name)

        # Enhance all the windows for a given audio file

        batch_e = sess.run(X_hat, feed_dict={X_n:chunk_signal})

        # Apply the hann window to the enhanced chunks
        batch_hann = batch_e * self.opts["hann_window"]

        # Combine the enhanced windows into a single audio file
        enhanced = np.zeros(self.opts["overlap_c"] * len(batch_e) + self.opts["overlap_c"])

        idx = 0
        for j in range(len(batch_hann)):
            enhanced[idx:idx + self.opts["cs"]] = enhanced[idx:idx + self.opts["cs"]] + batch_hann[j]
            idx += self.opts["overlap_c"]

        # Clip the joined windows and apply de-emphesis operation
        enhanced = enhanced[0:len(audio_signal)]
        enhanced = de_emph(enhanced)

        # Save the file
        # fn = files[i].split('/')[-1]
        # print(fn)
        # sf.write(save_path + '/' + fn, enhanced, 16000)
        return enhanced