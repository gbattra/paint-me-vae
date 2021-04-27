# Greg Attra
# 04/24/2021

encoder_config = {
   "input_shape": (128, 128, 3),
   "latent_dim": 50,
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
           "type": "conv2d",
           "n_filters": 128,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "conv2d",
           "n_filters": 256,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "flatten"
       }
   ]
}

decoder_config = {
   "latent_dim": 50,
   "layers": [
       {
           "type": "dense",
           "n_units": 8*8*256
       },
       {
           "type": "reshape",
           "shape": (8, 8, 256)
       },
       {
           "type": "conv2dt",
           "n_filters": 256,
           "filter_size": 3,
           "strides": 2
       },
       {
           "type": "conv2dt",
           "n_filters": 128,
           "filter_size": 3,
           "strides": 2
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
