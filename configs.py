# Greg Attra
# 04/24/2021

encoder_config = {
   "input_shape": (128, 128, 3),
   "latent_dim": 2,
   "layers": [
       {
           "type": "conv2d",
           "n_filters": 32,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "conv2d",
           "n_filters": 64,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "flatten"
       }
   ]
}

decoder_config = {
   "latent_dim": 2,
   "layers": [
       {
           "type": "dense",
           "n_units": 32*32*64
       },
       {
           "type": "reshape",
           "shape": (32, 32, 64)
       },
       {
           "type": "conv2dt",
           "n_filters": 64,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "conv2dt",
           "n_filters": 32,
           "filter_size": 3,
           "strides": 2
       }
   ]
}
