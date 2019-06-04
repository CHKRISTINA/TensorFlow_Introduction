'''
Args:
d: dimension of the embedding
beta: penalty parameter in matrix B of 2nd order objective
alpha: weighing hyperparameter for 1st order objective
nu1: L1-reg hyperparameter
nu2: L2-reg hyperparameter
K: number of hidden layers in encoder/decoder
n_units: vector of length K-1 containing #units in hidden layers
         of encoder/decoder, not including the units in the
         embedding layer
n_iter: number of sgd iterations for first embedding (const)
xeta: adam step size parameter
n_batch: minibatch size for Adam
actfn: activation function for hidden layer
modelfile: Files containing previous encoder and decoder models
weightfile: Files containing previous encoder and decoder weights
'''

class SSDNE():
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._K, self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)


network_embedding = SSDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=5,
                     n_units=[256, 120, 60, 10], n_iter=1000, xeta=0.01,
                     n_batch=500, actfn: 'relu',
                     modelfile=['./intermediate/enc_model.json',
                                './intermediate/dec_model.json'],
                     weightfile=['./intermediate/enc_weights.hdf5',
                                 './intermediate/dec_weights.hdf5'])
                                
                                
