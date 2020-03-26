# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-08-07 10:46:06
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-09-09 00:01:27
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import os
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sys
sys.path.append('../')
from utils import config


class BOW_DimReduction(object):

    def __init__(self, features_dim,
                 projection='PCA',
                 use_hashing=False,
                 remove_stop_words=False):
        assert isinstance(features_dim, int)
        assert features_dim > 0
        self.features_dim = features_dim
        self.projection = projection
        self.use_hashing = use_hashing
        self.remove_stop_words = remove_stop_words

    def extract_features(self, dataset):
        """Perform feature extraction.

        Args:
            dataset (Dataset): Dataset containing data and target
            projection (str, optional): Projection method

        Returns:
            np.array: X and Y matrices

        Raises:
            ValueError: If projection method is unknown
        """
        if self.remove_stop_words:
            stop_words = set(stopwords.words('english'))
            for i, text in enumerate(dataset.data):
                tokens = word_tokenize(text)
                text = [tk for tk in tokens if tk not in stop_words]
                dataset.data[i] = " ".join(text)

        if self.use_hashing:
            vectorizer = HashingVectorizer(stop_words='english',
                                           alternate_sign=False,
                                           n_features=self.features_dim)
            X = vectorizer.transform(dataset.data).todense()

        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
            X = vectorizer.fit_transform(dataset.data)

            if self.projection == 'PCA':
                X = PCA(n_components=self.features_dim).fit_transform(X.todense())

            elif self.projection == 'SparseRandom':
                # reduce dimensionality of the tf-idf features
                rp = SparseRandomProjection(n_components=self.features_dim)
                X = rp.fit_transform(X).todense()
            else:
                raise ValueError("Unknown projection method: {}".format(self.projection))

        y = dataset.target

        return X, y


class BOW_TopicModel(object):

    def __init__(self, nb_topics=10):
        """Class constructor.

        Args:
            nb_topics (int, optional): Number of topics for LDA.
        """
        assert isinstance(nb_topics, int)
        assert nb_topics > 0
        self.nb_topics = nb_topics

    def extract_features(self, dataset):
        """Extract features using Bag-of-Words and then LDA (topic models).
        Each document is represented as a n-dimensional vector, which
        represent the amount of information about each one of the 'n'
        topics.

        Args:
            dataset (Dataset): Dataset object containing all documents.

        Returns:
            X, y (np.array): Features and labels.
        """
        vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     max_df=0.5,
                                     stop_words='english')
        bow = vectorizer.fit_transform(dataset.data)

        self.lda_model = LatentDirichletAllocation(n_components=self.nb_topics,
                                                   random_state=0)

        X = self.lda_model.fit_transform(bow)
        y = dataset.target

        return X, y


class BERT(object):

    """ Use pre-trained BERT to extract feature from text. BERT only extracts
    features from sentence at most 512-token long. Longer texts are broken
    into sentences and then the mean embedding of all sentences is taken
    as embedding for the entire document. This code requires the pytorch_transformers
    package. It thas to be installed with pip. See:
    https://huggingface.co/pytorch-transformers/index.html
    """

    def __init__(self, sentence_len=100):

        self.sentence_len = sentence_len
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def extract_features(self, dataset):
        """Extract features using pre-trained BERT model.

        Args:
            dataset (dataset): Dataset object with raw/pre-processed text

        Returns:
            np.array: Features extracted and labels.
        """
        # check if features have aready been computed. save a lot of time
        fname = '{}/BERT_{}_sentlen_{}.pkl'.format(config.path_to_precomputed,
                                                   dataset.name, self.sentence_len)
        if os.path.exists(fname):
            print('Feature already extracted. Loading it ...')
            with open(fname, 'rb') as fh:
                X, y = pickle.load(fh)
            return X, y

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        corpus_embeddings = list()
        for d_i, doc in enumerate(dataset.data):
            # print('BERT: {} of {}'.format(d_i, len(dataset.data)))
            doc = doc.replace('\n', ' ')
            doc = " ".join(doc.split())  # remove extra spaces
            doc_tokens = doc.split(' ')

            # create batch: batch_size is equals to the number of
            # sentences (defined by self.setence_len)
            doc_embeddings = list()
            pivot = 0
            while pivot < len(doc_tokens):

                # get sentence
                end_sentence = np.minimum(pivot + self.sentence_len, len(doc_tokens))
                sentence = ' '.join(doc_tokens[pivot:end_sentence])

                # add tags required by BERT
                marked_text = "[CLS] " + sentence + " [SEP]"
                # tokenize the sentence (break it into tokens/words)
                tokenized_text = self.tokenizer.tokenize(marked_text)
                # get the token code (ID) to each token in the sentence
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                pivot += self.sentence_len

                # BERT is trained on pair of sentences (this vector says which
                # sentence each token belongs to). this is a binary vector
                segments_ids = [1] * len(tokenized_text)

                # must be torch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])

                # run model forward
                with torch.no_grad():
                    try:
                        token_embeddings, _ = self.model(tokens_tensor, segments_tensors)
                    except Exception:
                        print('**Failed to provide embeddings for: \n{}'.format(sentence))

                sentence_embedding = torch.mean(token_embeddings, 1)
                doc_embeddings.append(sentence_embedding.numpy())
            np_doc_embeddings = np.array(doc_embeddings)
            corpus_embeddings.append(np_doc_embeddings.mean(axis=0))

        X = np.squeeze(np.array(corpus_embeddings))  # remove extra dimension
        y = dataset.target

        # store computed features to save time in the next experiments
        # just load pre_computed features
        with open(fname, 'wb') as fh:
            pickle.dump([X, y], fh)

        return X, y
