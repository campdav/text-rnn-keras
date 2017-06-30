# text-rnn-keras
Tutorial: Multi-layer Recurrent Neural Networks (LSTM) for text models in Python using Keras.

Before going through this tutorial, I suggest to read the very very good blog note from Andrej Karpathy: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

# Requirements
- [Tensorflow 1.1.0rc0](http://www.tensorflow.org)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Keras 2.0.5](https://keras.io/)

# Training Tutorial
The project has **train.py** script.

The script generate and save locally a model and text features.

Just run it using python train.py
some items can be modified by editing the script:
- **data_dir** = 'data/Artistes_et_Phalanges-David_Campion'# data directory containing input.txt
- **save_dir** = 'save' # directory to store models
- **rnn_size** = 128 # size of RNN
- **batch_size** = 30 # minibatch size
- **seq_length** = 15 # sequence length
- **num_epochs** = 8 # number of epochs
- **learning_rate** = 0.001 #learning rate
- **sequences_step** = 1 #step to create sequences

note: 8 epochs is cleraly not enough. Should be above 25.

# Text generation Tutorial
The project has a **generate,py** script.

Just run it using python generate.py
some items can be modified by editing the script:
- **save_dir** = 'save' # directory where model is stored
- **batch_size** = 30 # minibatch size
- **seq_length** = 25 #  sequence length
- **words_number** = 400 #number of words to generate
- **seed_sentences** = "Il y a" #sentence for seed generation


# additional notes
The project comes with two types of input:
- __data/tinyshakespeare/input.txt__:
  - a small condensate of Shakespeare books
- __data/Artistes_et_Phalanges-David_campion/input.txt__:
  - The complete text of a french fantasy book "Artistes et Phalanges", by David Campion
  - This file book is under the following licence: Licence Creative Commons [CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)
