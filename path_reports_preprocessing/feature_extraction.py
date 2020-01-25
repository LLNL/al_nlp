import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt


def extract_tfidf(list_of_texts, max_features=5000):

    vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.95, min_df=2)

    # tokenize and build vocab
    vectorizer.fit(list_of_texts)

    list_of_transf = []
    for text in list_of_texts:
        # encode document
        vector = vectorizer.transform([text]).todense()
        list_of_transf.append(vector.copy())

    return list_of_transf


def extract_hashingvectorizer(list_of_texts, n_features=50):

    # create the transform
    vectorizer = HashingVectorizer(n_features=20)

    # there is not need to fit()

    list_of_transf = []

    # encode document
    for text in list_of_texts:
        # encode document
        vector = vectorizer.transform([text])  # .todense()
        list_of_transf.append(vector.copy())

    return list_of_transf


if __name__ == '__main__':

    path_to_data = 'list_of_reports.pkl'
    with open(path_to_data, 'rb') as fh:
        texts = pickle.load(fh)

    feats = extract_tfidf(texts)
    feat_mat = np.array(feats)
    feat_mat = np.squeeze(feat_mat)
    print(feat_mat.shape)

    svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
    # svd = PCA(n_components=2, random_state=42)
    svd.fit(feat_mat)
    proj = svd.transform(feat_mat)

    plt.plot(proj[:, 0], proj[:, 1], 'bo')
    plt.show()
