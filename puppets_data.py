import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import os
import scipy.misc
import matplotlib.image as mpimg


def read_img(fpath, grayscale=False, bbox=None):
    if grayscale:
        img = ImageOps.grayscale(Image.open(fpath))
    else:
        img = Image.open(fpath)
    if bbox is not None:
        return np.asarray(img.crop(bbox).reduce(2))
    else:
        return np.asarray(img.reduce(2))


def do_pca(X, n_pca):
    # print('Applying PCA')
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X)
    # print('explained_variance_ratio:', pca.explained_variance_ratio_)
    # print('sum(explained_variance_ratio):', np.sum(pca.explained_variance_ratio_))
    # print('singular_values:', pca.singular_values_)
    X = pca.fit_transform(X)
    return X

def puppets_data(dirpath, prefix='s1', n=None, bbox=None,
                 grayscale=False, normalize=False, n_pca=100):

    X = []
    labels = []
    fNames = []
    for fname in sorted(os.listdir(dirpath)):
        if prefix in fname:
            fNames.append(fname)

    if n is not None:
        fNames = fNames[:n]

    for fname in fNames:
        X_k = read_img(dirpath + '/' + fname, bbox=bbox, grayscale=grayscale)
        X.append(X_k.T.flatten())
        labels.append(int(fname.split('.')[0].split('_')[1]) - 100000)

    # img_shape = X.shape
    X = np.array(X)
    labels = np.array(labels)[:, np.newaxis] - 1
    labelsMat = np.concatenate([labels, labels], axis=1)

    if normalize:
        X = X - np.mean(X, axis=0)[np.newaxis, :]
        X = X / (np.std(X, axis=0)[np.newaxis, :] + 1e-12)

    if n_pca:
        X_new = do_pca(X, n_pca)
    else:
        X_new = X

    X_new = X_new / np.max(np.abs(X_new))
    print('X.shape = ', X_new.shape)
    return X_new, labelsMat, X #, img_shape
