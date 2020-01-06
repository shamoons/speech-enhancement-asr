import numpy as np
# from .se-vcae import encoder

class SeVCAE:
    def __init__(self, model_name):
        self.model_name = model_name
        self.opts = {
            "z_dim": 330,
            "peek": 200,
            "cs": 600
        }

    def pre_emph(self, x, coeff=0.95):
        x0 = np.reshape(x[0], [1,])
        diff = x[1:] - coeff * x[:-1]
        concat = np.concatenate([x0, diff], axis=0)
        return concat