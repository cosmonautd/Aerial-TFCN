import os
import cv2
import numpy as np
import sklearn.model_selection
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels
from sklearn import svm
import pickle

import sys
sys.path.append('../../')

import toolbox

try:
    import google.colab
    running = 'colab'
    colab = [running]
except:
    running = 'local'
    colab = []

if running in colab:
    from google.colab import drive, auth
    drive.mount('/content/drive')
    root = 'drive/My Drive/Colab Notebooks/TFCN'
else: 
    root = '.'

ap = argparse.ArgumentParser()
ap.add_argument('--test', type=int, default=1, required=False, help='Select a map to use as test')
ap.add_argument('--use', type=int, default=7, required=False, help='Select how many images to use')
args = ap.parse_args()

X_path = os.path.join(root, '../../dataset/X')
Y_path = os.path.join(root, '../../dataset/Y')

# Criação de diretório para armazenar a SVM treinada
W_path = os.path.join(root, 'svm')
if not os.path.exists(W_path):
    os.makedirs(W_path)

X_img = list()
Y_img = list()

resolution = 15

# Obtenção das imagens de entrada e rótulos de atravessabilidade
for i, imagename in enumerate(sorted(os.listdir(X_path))[:args.use]):
    if imagename.endswith('.jpg'):
        x = toolbox.imread(os.path.join(X_path, imagename), color='rgb')
        y = toolbox.imread(os.path.join(X_path, imagename), color='grayscale')
        X_img.append(x)
        Y_img.append(y)
        
def extract_features(region):
    features = []
    # 1) The average value r of the red content in the image
    features.append(np.mean(region[:,:,0]))
    # 2) The average value g of the green content in the image.
    features.append(np.mean(region[:,:,1]))
    # 3) The average value b of the blue content in the image.
    features.append(np.mean(region[:,:,2]))
    # 4) The mean m of the gray image.
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
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
cross_val.get_n_splits(X_img)

# Total de amostras no dataset original
N = len(X_img)

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X_img):

    if test_index + 1 != args.test:
        continue
    
    filename = os.path.join(W_path, 'svm-%d-%02d.pkl' % (args.use, test_index + 1))

    if not os.path.exists(filename):

        # Extração das regiões e rótulos para treinamento

        X_train = list()
        Y_train = list()
        X_test = list()
        Y_test = list()

        for i in train_index:
            x = X_img[i]
            y = Y_img[i]
            grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
            for g in grid:
                [tlx, tly, r] = g
                region = x[tly:tly+r, tlx:tlx+r]
                if region.shape != (resolution,resolution,3): continue
                # Extrair features
                features = extract_features(region)
                # Definir output
                output = np.mean(y[tly:tly+r, tlx:tlx+r])/255
                X_train.append(features)
                Y_train.append(output)

        X_train = np.array(X_train).astype(np.float32)
        Y_train = np.array(Y_train).astype(np.float32)

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

        X_val = X_train[:int(0.2*len(X_train))]
        Y_val = Y_train[:int(0.2*len(Y_train))]

        X_train = X_train[int(0.2*len(X_train)):]
        Y_train = Y_train[int(0.2*len(Y_train)):]

        # Treino da SVM
        clf = svm.SVR()
        clf.fit(X_train, Y_train)

        # Salvar SVM
        pickle.dump(clf, open(filename, 'wb'))

    # Carregar do disco
    clf = pickle.load(open(filename, 'rb'))

    # # Teste
    # y_img = np.zeros((1000,1000))
    # x = X_img[test_index[0]]
    # grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
    # for g in grid:
    #     [tlx, tly, r] = g
    #     region = x[tly:tly+r, tlx:tlx+r]
    #     if region.shape != (resolution,resolution,3): continue
    #     features = extract_features(region)
    #     traversability = clf.predict([features])[0]
    #     y_img[tly:tly+r, tlx:tlx+r] = (y_img[tly:tly+r, tlx:tlx+r] + traversability)/2
    
    # y_img = (y_img - np.min(y_img))/np.max(y_img)
    # toolbox.show_image([y_img])
    
    # Find optimal c based on training data

    import torch
    import graphmapx
    import itertools

    positive_keypoints_path = os.path.join(root, '../../dataset/kp/')
    negative_keypoints_path = os.path.join(root, '../../dataset/kn/')

    print('Optimizing %d...' % (args.test))

    T = np.zeros((len(train_index), 125, 125))
    k = 0

    for i in train_index:
        x = X_img[i]
        y_img = np.zeros((1000,1000))
        grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
        for g in grid:
            [tlx, tly, r] = g
            region = x[tly:tly+r, tlx:tlx+r]
            if region.shape != (resolution,resolution,3): continue
            features = extract_features(region)
            traversability = clf.predict([features])[0]
            y_img[tly:tly+r, tlx:tlx+r] = (y_img[tly:tly+r, tlx:tlx+r] + traversability)/2
        
        y_img = (y_img - np.min(y_img))/np.max(y_img)
        y = cv2.resize(y_img, (125, 125))
        y = y.astype(np.float32)
        T[k,:,:] = y
        k += 1

    def optimize():

        def func(c, r=8):

            exp = list()
            
            for i in range(len(T)):

                positive_keypoints = toolbox.imread(os.path.join(positive_keypoints_path, 'aerial%02d.jpg' % (train_index[i]+1)))
                negative_keypoints = toolbox.imread(os.path.join(negative_keypoints_path, 'aerial%02d.jpg' % (train_index[i]+1)))

                t_matrix = T[i]
                t_matrix = np.squeeze(t_matrix)
                t_matrix = (t_matrix - np.min(t_matrix))/np.max(t_matrix)

                gt = toolbox.imread(os.path.join(Y_path, 'aerial%02d.jpg' % (train_index[i]+1)))
                grid = toolbox.grid_list(gt, r)

                router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
                G = router.tm2graph(t_matrix)

                keypoints = graphmapx.get_keypoints(positive_keypoints, grid)
                combinations = list(itertools.combinations(keypoints, 2))

                for counter in range(len(combinations)):

                    (s, t) = combinations[counter]

                    path, found = router.route(G, s, t, t_matrix)

                    score = toolbox.score(path, gt, r)

                    results = dict()
                    results['path_existence'] = True
                    results['path_found'] = found
                    results['path_score'] = score if found else 0.0

                    exp.append(results)

                keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
                combinations = list(itertools.combinations(keypoints, 2))

                for counter in range(len(combinations)):

                    (s, t) = combinations[counter]

                    path, found = router.route(G, s, t, t_matrix)

                    results = dict()
                    results['path_existence'] = False
                    results['path_found'] = found
                    results['path_score'] = 1.0 if not found else 0.0

                    exp.append(results)
                
            score = list()
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for sample in exp:
                score.append(sample['path_score'])
                if sample['path_existence'] and sample['path_found']:
                    true_positive += 1
                elif not sample['path_existence'] and not sample['path_found']:
                    true_negative += 1
                elif not sample['path_existence'] and sample['path_found']:
                    false_positive += 1
                elif sample['path_existence'] and not sample['path_found']:
                    false_negative += 1
            
            recall = true_positive/(true_positive+false_negative)
            accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)

            frdr = recall
            irdr = true_negative/(true_negative + false_positive)
            fdr  = accuracy

            score_mean = np.mean(score)
            score_std = np.std(score)

            print('c: %.3f, score: %.3f, func: %.3f' % (c, score_mean, 1 - score_mean))

            return 1 - score_mean

        # c0 = 0.1
        # ret = scipy.optimize.minimize(func, c0, method='Powell')
        # c = ret.x[0]

        res = scipy.optimize.minimize_scalar(func, bounds=(0.0, 1.0), method='bounded',
                    options={'xatol': 0.01, 'maxiter': 20})
        
        c = res.x

        return c
    
    runs = [optimize() for _ in range(1)]

    c = np.median(runs)
    std = np.std(runs)

    print(c)
    
    C_file = os.path.join(W_path, 'c_%d_%02d.hdf5' % (args.use, test_index + 1))
    np.savetxt(C_file, np.array([c]), delimiter=',')