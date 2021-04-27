### Environment
* Ubuntu Desktop 2020
* IDE: PyCharm

### Dependencies
* `tensorflow`
* `numpy`
* `matplotlib`
* `pillow`

### Executables

* `train.py` - trains the specified VAE
    - usage:
        - `python train.py paintings`
        - `python train.py thumbnails`
    
* `paint.py` - paints an image using the paintings and faces VAEs
    - usage:
        - `python train.py <path_to_image> portrait_to_painting`
            - paints the provided face picture
        - `python train.py <path_to_image> painting_to_portrait`
            - converts a painting to a face picture