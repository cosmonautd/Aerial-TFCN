import os
import time
import argparse

import cv2 as cv
import numpy as np
import sklearn.model_selection

import tfcnmodel
import toolbox

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = "."

ap = argparse.ArgumentParser()
ap.add_argument("--rgb", help="Use TFCN-RGB", action="store_true")
ap.add_argument("--grayscale", help="Use TFCN-G", action="store_true")
ap.add_argument(
    "--test", type=int, default=1, required=False, help="Select a map to use as test"
)
ap.add_argument("--outliers", help="Include outliers", action="store_true")
args = ap.parse_args()

if args.rgb:
    COLOR = "rgb"
    WEIGHTS_DIR = "rgb"
elif args.grayscale:
    COLOR = "grayscale"
    WEIGHTS_DIR = "grayscale"
else:
    print("Please select a model using --rgb or --grayscale")
    exit()

X_path = os.path.join(root, "dataset/X")
Y_path = os.path.join(root, "dataset/Y")
W_path = os.path.join(
    root, "weights", WEIGHTS_DIR, "all" if args.outliers else "no outliers"
)

output_path__ = os.path.join(root, "results/maps")
if not os.path.exists(output_path__):
    os.makedirs(output_path__)

X = list()
Y = list()

n_samples = 8 if args.outliers else 7

for i, imagename in enumerate(sorted(os.listdir(X_path))[:n_samples]):
    if imagename.endswith(".jpg"):
        x = toolbox.imread(os.path.join(X_path, imagename), color=COLOR)
        y = toolbox.imread(os.path.join(Y_path, imagename), color="grayscale")
        y = y / 255
        X.append(x)
        Y.append(y)

X = np.array(X).astype(np.float32)
Y = np.expand_dims(np.array(Y), 3).astype(np.float32)

cross_val = sklearn.model_selection.LeaveOneOut()
cross_val.get_n_splits(X)

N = len(X)

TIME = []
LOSS = []
MSE = []

# Leave One Out
for train_index, test_index in cross_val.split(X):
    if test_index + 1 != args.test:
        continue

    val_index = (test_index + 1) % N
    train_index = np.array([x for x in train_index if x not in val_index])

    x_train, x_test, x_val = X[train_index], X[test_index], X[val_index]
    y_train, y_test, y_val = Y[train_index], Y[test_index], Y[val_index]

    if args.grayscale:
        x_test = np.expand_dims(x_test, 3)

    data = tfcnmodel.Data(None, None, None, None, x_test, y_test)
    dims = data.x_test.shape[1:]

    tfcn = tfcnmodel.TFCN(dims)
    tfcn.load(wp=os.path.join(W_path, "weights_%02d.hdf5" % (test_index + 1)))

    test_loss = tfcn.loss(data.x_test, data.y_test)

    time__ = time.time()

    y = tfcn.predict(data.x_test)
    y = np.squeeze(y[0])

    y = (y - np.min(y)) / np.max(y)

    time__ = time.time() - time__

    y__ = np.squeeze(data.y_test[0])

    toolbox.save_image(
        os.path.join(
            output_path__, "tfcn-output-%d-%s-%02d.jpg" % (N, COLOR, test_index + 1)
        ),
        [y],
    )

    y = cv.resize(y, (125, 125))
    y__ = cv.resize(y__, (125, 125))

    TIME.append(time__)
    LOSS.append(test_loss)
    MSE.append((((y - y__) ** 2).mean(axis=None)))

for M in MSE:
    print("  %.3f" % M)
