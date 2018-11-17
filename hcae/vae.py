from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Dropout
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers.merge import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l1, l2

from tybalt.utils.base import VAE, BaseModel

class Hcae(VAE):
    def __init__(self, original_dim, latent_dim, batch_size=50, epochs=50,
                 learning_rate=0.0005, kappa=1, epsilon_std=1.0,
                 beta=K.variable(0), loss='binary_crossentropy',
                 verbose=True):
        VAE.__init__(self)
        self.model_name = 'Tybalt'
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.epsilon_std = epsilon_std
        self.beta = beta
        self.loss = loss
        self.verbose = verbose
        pass

    def _build_encoder_layer(self):
        pass

    def _build_decoder_layer(self):
        pass

    def _compile(self):
        pass

    def _connect_layers(self):
        pass

    def train_vae(self):
        pass
