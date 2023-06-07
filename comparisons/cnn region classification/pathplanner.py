import os
import time
import argparse
import itertools
import multiprocessing

import cv2
import numpy as np
import sklearn.model_selection

import graphmapx

import sys

sys.path.append("../../")
import toolbox

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Leitura dos argumentos de linha de comando
ap = argparse.ArgumentParser()
ap.add_argument(
    "--test", type=int, default=1, required=False, help="Select a map to use as test"
)
ap.add_argument(
    "--use", type=int, default=7, required=False, help="Select how many images to use"
)
args = ap.parse_args()

try:
    from google.colab import drive, auth

    drive.mount("/content/drive")
    root = "drive/My Drive/Colab Notebooks/TFCN"
except:
    root = "."

resolution = 15
if args.use == 8:
    classes = ["road", "building", "grass", "vegetation", "vehicle", "sand"]
    cost_vector = [-100, 200, -60, 200, 150, -20]
elif args.use == 7:
    classes = ["road", "building", "grass", "vegetation", "vehicle"]
    cost_vector = [-100, 200, -60, 200, 150]

W_path = os.path.join(root, "weights%d" % (args.use))
X_path = os.path.join("..", "..", "dataset", "X")
Y_path = os.path.join("..", "..", "dataset", "Y")
KP_path = os.path.join("..", "..", "dataset", "kp")
KN_path = os.path.join("..", "..", "dataset", "kn")

# Criação de diretório para armazenar os caminhos calculados
P_path = os.path.join(root, "paths%d" % (args.use))
if not os.path.exists(P_path):
    os.mkdir(P_path)

X = list()
Y = list()
X_colored = list()

for imagename in sorted(os.listdir(X_path)[: args.use]):
    if imagename.endswith(".jpg"):
        x = toolbox.imread(os.path.join(X_path, imagename), color="rgb")
        x_colored = toolbox.imread(os.path.join(X_path, imagename), color="rgb")
        y = toolbox.imread(os.path.join(Y_path, imagename), color="grayscale")
        X.append(x)
        X_colored.append(x_colored)
        y = cv2.resize(y, (125, 125)) / np.max(y)
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

    # Treinamento da CNN
    # Função para construção da CNN
    def build_cnn():
        # Organização sequencial de camadas
        model = Sequential()
        model.add(
            Conv2D(
                filters=6,
                kernel_size=5,
                padding="same",
                activation="tanh",
                input_shape=(resolution, resolution, 3),
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=50, kernel_size=5, padding="same", activation="tanh"))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(len(classes), activation="tanh"))

        # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
        model.compile(loss="mse", optimizer="adam")
        return model

    # Construção da CNN
    CNN = build_cnn()

    # Pesos
    weights_path = os.path.join(W_path, "weights_%02d.hdf5" % (test_index + 1))

    # Carregamento da melhor combinação de pesos obtida durante o treinamento
    CNN.load_weights(weights_path)

    thresholds = np.loadtxt(
        os.path.join(W_path, "thresholds-%d.txt" % (test_index + 1)), dtype=float
    )

    # Teste
    y_img = np.zeros((1000, 1000))
    x = toolbox.imread(
        os.path.join(X_path, "aerial%02d.jpg" % (test_index + 1)), color="rgb"
    )
    grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
    for g in grid:
        [tlx, tly, r] = g
        region = x[tly : tly + r, tlx : tlx + r]
        if region.shape != (resolution, resolution, 3):
            continue
        y = CNN.predict(np.array([region]))[0]
        for k, _ in enumerate(y):
            d = y[k] - thresholds[k]
            if d > 0:
                y[k] = 0.5 + min(0.5, 1.5 * d)
            elif d < 0:
                y[k] = 0.5 + max(-0.5, 1.5 * d)
            else:
                y[k] = 0.5
        c_unknown = 200
        cost = y @ cost_vector + c_unknown
        y_img[tly : tly + r, tlx : tlx + r] = (
            y_img[tly : tly + r, tlx : tlx + r] + cost
        ) / 2

    y_img = cv2.resize(y_img, (125, 125))

    T = y_img - np.min(y_img)
    T = T.astype(np.float32)

    GT = toolbox.imread(
        os.path.join(Y_path, "aerial%02d.jpg" % (test_index[0] + 1)), color="rgb"
    )

    keypoints_image = toolbox.imread(
        os.path.join(KP_path, "aerial%02d.jpg" % (test_index[0] + 1)), color="rgb"
    )
    grid = toolbox.grid_list(x, 8)
    keypoints = graphmapx.get_keypoints(keypoints_image, grid)

    c = c_unknown

    router = graphmapx.RouteEstimator(r=8, c=c, grid=grid)
    G = router.tm2graph(T)

    output_path = "paths%d/%s" % (args.use, "aerial%02d" % (test_index[0] + 1))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):

        path, found = router.route(G, s, t, T)

        score__ = toolbox.score(path, GT, 8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (10, 40)
        fontScale = 1
        lineType = 2

        fontColor = (255, 0, 0) if score__ < 0.7 else (0, 255, 0)

        path_image = toolbox.draw_path(
            X_colored[test_index[0]], path, found=found, color=fontColor
        )
        cv2.putText(
            path_image,
            "Score: %.2f" % (score__),
            topLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )

        toolbox.save_image(
            os.path.join(output_path, "path-%d.jpg" % (counter + 1)), [path_image]
        )
