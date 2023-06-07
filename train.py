import os
import cv2 as cv
import numpy as np
import sklearn.model_selection
import argparse
import scipy.optimize
import skimage.transform

import tfcnmodel
import toolbox

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

try:
    import google.colab

    running = "colab"
    colab = [running]
except:
    running = "local"
    colab = []

if running in colab:
    from google.colab import drive, auth

    drive.mount("/content/drive")
    root = "drive/My Drive/Colab Notebooks/TFCN + Conv2DTranspose (keras)"
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = "."

ap = argparse.ArgumentParser()
ap.add_argument("--rgb", help="Use TFCN-RGB", action="store_true")
ap.add_argument("--grayscale", help="Use TFCN-G", action="store_true")
ap.add_argument("--hsv", help="Use TFCN-HSV", action="store_true")
ap.add_argument("--rgbh", help="Use TFCN-RGBH", action="store_true")
ap.add_argument(
    "--test", type=int, default=1, required=False, help="Select a map to use as test"
)
ap.add_argument("--outliers", help="Include outliers", action="store_true")
ap.add_argument("--overwrite", help="Overwrite weights", action="store_true")
args = ap.parse_args()

if args.rgb:
    COLOR = "rgb"
    DIR = "rgb"
elif args.grayscale:
    COLOR = "grayscale"
    DIR = "grayscale"
elif args.hsv:
    COLOR = "hsv"
    DIR = "hsv"
elif args.rgbh:
    COLOR = "rgbh"
    DIR = "rgbh"

X_path = os.path.join(root, "dataset/X")
Y_path = os.path.join(root, "dataset/Y")

# Criação de diretório para armazenar os pesos treinados
W_path = os.path.join(root, "weights", DIR, "all" if args.outliers else "no outliers")
if not os.path.exists(W_path):
    os.makedirs(W_path)

# Criação de diretório para armazenar os gráficos de perda
L_path = os.path.join(root, "losses", DIR, "all" if args.outliers else "no outliers")
if not os.path.exists(L_path):
    os.makedirs(L_path)

X = list()
Y = list()

n_samples = 8 if args.outliers else 7

for i, imagename in enumerate(sorted(os.listdir(X_path))[:n_samples]):
    if imagename.endswith(".jpg"):
        x = toolbox.imread(os.path.join(X_path, imagename), color=COLOR)
        y = toolbox.imread(os.path.join(Y_path, imagename), color="grayscale")
        # y = cv.resize(y, (125, 125))/np.max(y)
        y = y / 255
        X.append(x)
        Y.append(y)

X = np.array(X).astype(np.float32)
Y = np.expand_dims(np.array(Y), 3).astype(np.float32)

cross_val = sklearn.model_selection.LeaveOneOut()
cross_val.get_n_splits(X)

# Total de amostras no dataset original
N = len(X)

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X):

    if test_index + 1 != args.test:
        continue

    # Índice do conjunto de validação
    val_index = (test_index + 1) % N
    train_index = np.array([x for x in train_index if x not in val_index])

    x_train, x_test, x_val = X[train_index], X[test_index], X[val_index]
    y_train, y_test, y_val = Y[train_index], Y[test_index], Y[val_index]

    if args.grayscale:
        x_val = np.expand_dims(x_val, 3)
        x_train = np.expand_dims(x_train, 3)

    data = tfcnmodel.Data(x_train, y_train, x_val, y_val, None, None)
    dims = data.x_train.shape[1:]

    tfcn = tfcnmodel.TFCN(dims)

    W_file = os.path.join(W_path, "weights_%02d.hdf5" % (test_index + 1))
    if not os.path.exists(W_file) or args.overwrite:
        tfcn.train(data, wp=W_file)
        if running in colab:
            toolbox.visualize_training(tfcn.history, test_index + 1, L_path)
        else:
            toolbox.visualize_training(
                tfcn.history, test_index + 1, L_path, block=False
            )
    else:
        tfcn.load(wp=W_file)

    # Find optimal c based on training data

    import torch
    import graphmapx
    import itertools

    positive_keypoints_path = os.path.join(root, "dataset/kp/")
    negative_keypoints_path = os.path.join(root, "dataset/kn/")

    y = tfcn.predict(data.x_train)

    print("Optimizing %d by Brent's method" % (args.test))

    def optimize():
        def func(c, r=8):

            exp = list()

            T = tfcn.predict(data.x_train)

            for i in range(len(T)):

                positive_keypoints = toolbox.imread(
                    os.path.join(
                        positive_keypoints_path, "aerial%02d.jpg" % (train_index[i] + 1)
                    )
                )
                negative_keypoints = toolbox.imread(
                    os.path.join(
                        negative_keypoints_path, "aerial%02d.jpg" % (train_index[i] + 1)
                    )
                )

                t_matrix = T[i]
                t_matrix = skimage.transform.resize(
                    t_matrix, (125, 125), anti_aliasing=True
                )
                t_matrix = np.squeeze(t_matrix)
                t_matrix = (t_matrix - np.min(t_matrix)) / np.max(t_matrix)

                gt = toolbox.imread(
                    os.path.join(Y_path, "aerial%02d.jpg" % (train_index[i] + 1))
                )
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
                    results["path_existence"] = True
                    results["path_found"] = found
                    results["path_score"] = score if found else 0.0

                    exp.append(results)

                keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
                combinations = list(itertools.combinations(keypoints, 2))

                for counter in range(len(combinations)):

                    (s, t) = combinations[counter]

                    path, found = router.route(G, s, t, t_matrix)

                    results = dict()
                    results["path_existence"] = False
                    results["path_found"] = found
                    results["path_score"] = 1.0 if not found else 0.0

                    exp.append(results)

            score = list()
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0

            for sample in exp:
                score.append(sample["path_score"])
                if sample["path_existence"] and sample["path_found"]:
                    true_positive += 1
                elif not sample["path_existence"] and not sample["path_found"]:
                    true_negative += 1
                elif not sample["path_existence"] and sample["path_found"]:
                    false_positive += 1
                elif sample["path_existence"] and not sample["path_found"]:
                    false_negative += 1

            recall = true_positive / (true_positive + false_negative)
            accuracy = (true_positive + true_negative) / (
                true_positive + true_negative + false_positive + false_negative
            )

            frdr = recall
            irdr = true_negative / (true_negative + false_positive)
            fdr = accuracy

            score_mean = np.mean(score)
            score_std = np.std(score)

            output = 1 - (2 * score_mean + frdr + 3 * irdr) / 6

            print(
                "c: %.3f, p: %.3f, f+: %.3f, f-: %.3f, output: %.3f"
                % (c, score_mean, frdr, irdr, output)
            )

            return output

        # c0 = 0.1
        # ret = scipy.optimize.minimize(func, c0, method='Powell')
        # c = ret.x[0]

        res = scipy.optimize.minimize_scalar(
            func,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 0.01, "maxiter": 20},
        )

        c = res.x

        return c

    runs = [optimize() for _ in range(1)]

    c = np.median(runs)
    std = np.std(runs)

    print("c = %.3f" % (c))

    C_file = os.path.join(W_path, "c_%02d.hdf5" % (test_index + 1))
    np.savetxt(C_file, np.array([c]), delimiter=",")
