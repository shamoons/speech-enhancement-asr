# # import sys
# # import os
# # from pathlib import Path

# # IMPORT_PATH = Path(__file__)
# # print('IMPORT_PATH', IMPORT_PATH)

# # sys.path.append(IMPORT_PATH)

# import tensorflow as tf
# import numpy as np
# import pickle

# from .deepxi.lib.dev.ResNet import ResNet
# from .deepxi.lib.dev.acoustic.feat import polar
# from .deepxi.lib.dev.acoustic.analysis_synthesis.polar import synthesis
# from .deepxi.lib.dev import optimisation

# class DeepXiNet:
#     def __init__(self):
#         print('Preparing graph...')

#         self.args = {
#             "d_in": 257,
#             "norm_type": "FrameLayerNorm",
#             "n_blocks": 40,
#             "d_out": 257,
#             "d_model": 256,
#             "d_f": 64,
#             "k_size": 3,
#             "max_d_rate": 16,
#             "f_s": 16000,
#             "T_w": 32,
#             "T_s": 16
#         }

#         self.args['N_w'] = int(self.args["f_s"]*self.args["T_w"]*0.001)
#         self.args['N_s'] = int(self.args["f_s"]*self.args["T_s"]*0.001)
#         self.args['NFFT'] = int(pow(2, np.ceil(np.log2(self.args['N_w']))))

#         self.args['stats'] = self.get_stats(self.args)

#         # RESNET
#         # noisy speech MS placeholder.
#         self.input_ph = tf.placeholder(
#             tf.float32, shape=[None, None, self.args["d_in"]], name='input_ph')
#         # noisy speech MS sequence length placeholder.
#         self.nframes_ph = tf.placeholder(
#             tf.int32, shape=[None], name='nframes_ph')
#         self.output = ResNet(self.input_ph, self.nframes_ph, self.args["norm_type"], n_blocks=self.args["n_blocks"],
#                              boolean_mask=True, d_out=self.args["d_out"], d_model=self.args["d_model"], d_f=self.args["d_f"], k_size=self.args["k_size"],
#                              max_d_rate=self.args["max_d_rate"])

#         # TRAINING FEATURE EXTRACTION GRAPH
#         # clean speech placeholder.
#         self.s_ph = tf.placeholder(tf.int16, shape=[None, None], name='s_ph')
#         # noise placeholder.
#         self.d_ph = tf.placeholder(tf.int16, shape=[None, None], name='d_ph')
#         # clean speech sequence length placeholder.
#         self.s_len_ph = tf.placeholder(tf.int32, shape=[None], name='s_len_ph')
#         # noise sequence length placeholder.
#         self.d_len_ph = tf.placeholder(tf.int32, shape=[None], name='d_len_ph')
#         # SNR placeholder.
#         self.snr_ph = tf.placeholder(tf.float32, shape=[None], name='snr_ph')
#         self.train_feat = polar.input_target_xi(self.s_ph, self.d_ph, self.s_len_ph,
#                                                 self.d_len_ph, self.snr_ph, self.args["N_w"], self.args["N_s"], self.args["NFFT"], self.args["f_s"], self.args['stats']['mu_hat'], self.args['stats']['sigma_hat'])

#         # INFERENCE FEATURE EXTRACTION GRAPH
#         self.infer_feat = polar.input(
#             self.s_ph, self.s_len_ph, self.args["N_w"], self.args["N_s"], self.args["NFFT"], self.args["f_s"])

#         # PLACEHOLDERS
#         # noisy speech placeholder.
#         self.x_ph = tf.placeholder(tf.int16, shape=[None, None], name='x_ph')
#         # noisy speech sequence length placeholder.
#         self.x_len_ph = tf.placeholder(tf.int32, shape=[None], name='x_len_ph')
#         # training target placeholder.
#         self.target_ph = tf.placeholder(
#             tf.float32, shape=[None, self.args["d_out"]], name='target_ph')
#         # keep probability placeholder.
#         self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
#         # training placeholder.
#         self.training_ph = tf.placeholder(tf.bool, name='training_ph')

#         # SYNTHESIS GRAPH
#         # if self.args.infer:
#         self.infer_output = tf.nn.sigmoid(self.output)
#         self.y_MAG_ph = tf.placeholder(
#             tf.float32, shape=[None, None, self.args["d_in"]], name='y_MAG_ph')
#         self.x_PHA_ph = tf.placeholder(
#             tf.float32, [None, None, self.args["d_in"]], name='x_PHA_ph')
#         self.y = synthesis(self.y_MAG_ph, self.x_PHA_ph,
#                            self.args["N_w"], self.args["N_s"], self.args["NFFT"])

#         ## LOSS & OPTIMIZER
#         self.loss = optimisation.loss(
#             self.target_ph, self.output, 'mean_sigmoid_cross_entropy', axis=[1])
#         self.total_loss = tf.reduce_mean(self.loss, axis=0)
#         self.trainer, _ = optimisation.optimiser(
#             self.total_loss, optimizer='adam', grad_clip=True)

#         # SAVE VARIABLES
#         self.saver = tf.compat.v1.train.Saver(max_to_keep=256)

#         # NUMBER OF PARAMETERS
#         self.args['params'] = (np.sum([np.prod(v.get_shape().as_list())
#                                     for v in tf.trainable_variables()]))


#     ## GET STATISTICS OF SAMPLE
#     def get_stats(self):
#         print('Loading sample statistics from pickle file...')
#         stats_file = 'lib/deepxi/data/3e_set/stats.p'
#         with open(stats_file, 'rb') as f:
#             stats = pickle.load(f)
#         return stats
