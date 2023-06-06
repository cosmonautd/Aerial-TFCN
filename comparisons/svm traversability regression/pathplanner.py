import os
import time
import argparse
import itertools
import multiprocessing
import pickle

import cv2 as cv
import numpy as np
import sklearn.model_selection

import sys
sys.path.append('../../')

import toolbox
import graphmapx

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("--test", type=int, default=1, required=False, help="Select a map to use as test")
ap.add_argument("--use", type=int, default=7, required=False, help="Select how many images to use")
args = ap.parse_args()

try:
    from google.colab import drive, auth
    drive.mount('/content/drive')
    root = 'drive/My Drive/Colab Notebooks/TFCN'
except: 
    root = '.'

W_path = os.path.join(root, 'svm')
X_path = os.path.join('../../', 'dataset', 'X')
Y_path = os.path.join('../../', 'dataset', 'Y')
KP_path = os.path.join('../../', 'dataset', 'kp')
KN_path = os.path.join('../../', 'dataset', 'kn')

# Criação de diretório para armazenar os caminhos calculados
P_path = os.path.join(root, 'paths%d' % (args.use))
if not os.path.exists(P_path):
    os.mkdir(P_path)

X = list()
Y = list()
X_colored = list()

for imagename in sorted(os.listdir(X_path)[:args.use]):
    if imagename.endswith('.jpg'):
        x = toolbox.imread(os.path.join(X_path, imagename), color='rgb')
        x_colored = toolbox.imread(os.path.join(X_path, imagename), color='rgb')
        y = toolbox.imread(os.path.join(Y_path, imagename), color='grayscale')
        X.append(x)
        X_colored.append(x_colored)
        y = cv.resize(y, (125, 125))/np.max(y)
        Y.append(y)

X = np.array(X).astype(np.float32)
Y = np.expand_dims(np.array(Y), 3).astype(np.float32)

resolution = 15

def extract_features(region):
    features = []
    # 1) The average value r of the red content in the image
    features.append(np.mean(region[:,:,0]))
    # 2) The average value g of the green content in the image.
    features.append(np.mean(region[:,:,1]))
    # 3) The average value b of the blue content in the image.
    features.append(np.mean(region[:,:,2]))
    # 4) The mean m of the gray image.
    gray_region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    mu = np.mean(gray_region)
    features.append(mu)
    # 5) The standard deviation σ of the gray image.
    features.append(np.array(np.std(gray_region)))
    # 6) The smoothness R of the gray image.
    H = np.histogram(gray_region, bins=256)[0] + 1
    total = np.sum(H)
    numerator = np.sum([(zi/total)*(zi - mu)**2 for zi in H])
    R = numerator/(1 + numerator)
    features.append(R)
    # 7) The third moment μ3 . The feature is a measurement of the skewness of a histogram.
    features.append(np.sum([(zi/total)*(zi - mu)**3 for zi in H]))
    # 8) The uniformity U.
    features.append(np.sum([(zi/total)**2 for zi in H]))
    # 9) The entropy e. The feature is a measurement of randomness for the all gray levels of the intensity histogram.
    e = - np.sum([(zi/total)*np.log2(zi/total) for zi in H])
    features.append(e)

    return features

cross_val = sklearn.model_selection.LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras no dataset original
N = len(X)

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X):

    if test_index + 1 != args.test:
        continue

    filename = os.path.join(W_path, 'svm-%d-%02d.pkl' % (args.use, test_index + 1))
    clf = pickle.load(open(filename, 'rb'))

    y = np.zeros((1000,1000))
    x = toolbox.imread(os.path.join(X_path, 'aerial%02d.jpg' % (test_index+1)), color='rgb')
    grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
    for g in grid:
        [tlx, tly, r] = g
        region = x[tly:tly+r, tlx:tlx+r]
        if region.shape != (resolution,resolution,3): continue
        features = extract_features(region)
        traversability = clf.predict([features])[0]
        y[tly:tly+r, tlx:tlx+r] = (y[tly:tly+r, tlx:tlx+r] + traversability)/2

    y = (y - np.min(y))/np.max(y)
    y = cv.resize(y, (125, 125))
    y = y.astype(np.float32)

    T = y
    GT = toolbox.imread(os.path.join(Y_path, 'aerial%02d.jpg' % (test_index[0]+1)), color='rgb')

    keypoints_image = toolbox.imread(os.path.join(KP_path, 'aerial%02d.jpg' % (test_index[0]+1)), color='rgb')
    grid = toolbox.grid_list(x, 8)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    C_file = os.path.join(w_path, 'c_%d_%02d.hdf5' % (args.use, test_index + 1))
    c = 1.25*np.loadtxt(C_file).item()

    router = graphmapx.RouteEstimator(r=8, c=c, grid=grid)
    G = router.tm2graph(T)

    output_path = 'paths%d/%s' % (args.use, 'aerial%02d' % (test_index[0]+1))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, T)

        score__ = toolbox.score(path, GT, 8)

        font                   = cv.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText    = (10, 40)
        fontScale              = 1
        lineType               = 2

        fontColor = (255,0,0) if score__ < 0.7 else (0,255,0)

        path_image = toolbox.draw_path(X_colored[test_index[0]], path, found=found, color=fontColor)
        cv.putText(path_image, 'Score: %.2f' % (score__),
                    topLeftCornerOfText, font, fontScale, fontColor, lineType)

        toolbox.save_image(os.path.join(output_path, 'path-%d.jpg' % (counter+1)), [path_image])