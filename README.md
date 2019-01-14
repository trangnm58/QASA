**QASA Document Retriever Model**
================

Installation
---------------

This program was developed using Python version 3.5 and was tested on Linux and Windows system.
We recommend using Anaconda 4.2 for installing **Python 3.5** as well as **numpy**, although you can install them by other means.

Other requirements:

1. **Tensorflow** GPU 1.4.1:

> pip install tensorflow-gpu==1.4.1

If GPU is not available, tensorflow CPU can be used instead:
> pip install tensorflow==1.4.1

2. **Keras** 2.2.2:
> pip install Keras==2.2.2

3. **Sklearn** 0.20.0:
> conda install scikit-learn==0.20.0

4. **NLTK** 3.3:
> conda install nltk==3.3

5. **SpaCy** 2.0.16:
> conda install spacy==2.0.16

and spaCy model:
> python -m spacy download en_core_web_lg

Usage
---------

#### **Evaluating pre-trained models**
Command:

    python qasa.py -h
    usage: qasa.py [-h] [-eval] [-es] [-e EPOCH] [-k K_NEG] [-c]
               model dataset train_set

    Train new models/Evaluate pre-trained models.

    positional arguments:
      model                 the name of the model
      dataset               the name of the dataset that the model will be trained
                            on, e.g: quasart
      train_set             e.g: train_pairwise.pickle

    optional arguments:
      -h, --help            show this help message and exit
      -eval, --evaluate     evaluate existing model
      -es, --early_stopping
                            use early stopping
      -e EPOCH, --epoch EPOCH
                            number of epochs to train or maximum epoch when early
                            stopping
      -k K_NEG, --k_neg K_NEG
                            number of negative examples to be sampled
      -c, --is_continue     continue to train or not

> **Note:**
> - "model" is the direct child folder within the "trained_models/" folder
> - "dataset" is the direct child folder within the "data/" folder
> - "train_set" is the file within \<dataset\>. It is generated when building new training dataset in training phase.

Example: Evaluating the model trained on the QUASAR-T dataset using its test set.

    python qasa.py qasa_neural quasart train_pairwise.pickle -eval

#### **Training new models**
[**Step 0** - Download the pre-trained word embedding model]

Any pre-trained word embedding model can be used but it must use Glove format as our provided word embedding model, which we highly recommend to be used. You can download the model file named "**wiki.en.vec**" at this link: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

After the file was downloaded, place it within the "**data/w2v_model**" folder.

[**Step 1** - Prepare the converted data]

The converted data will be prepared when running the follow command:

    python build_data.py <dataset> <word_embedding> 

Example: Building converted data for QUASAR-T dataset with FastText embeddings
  
    python build_data.py quasart data/w2v_model/wiki.en.vec

[**Step 2** - Prepare training data]

The training data will be prepared when running the follow command:

    python dataset.py <dataset> <data_file> 

Example: Prepare training data for QUASAR-T dataset.

    python dataset.py quasart train_pairwise

[**Step 3** - Training]

Example: Training a new model on QUASAR-T dataset with early stopping option and sample 50 negative answers

    python qasa.py qasa_neural quasart train_pairwise.pickle -es -e 5 -k 50


Contacts
------------

If you have any questions or problems, please email **pagenguyen219@gmail.com** (Trang M. Nguyen)
