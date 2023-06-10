import os
import time
import argparse
import itertools

import cv2 as cv
import numpy as np
import sklearn.model_selection
import skimage.transform
import tqdm

import tfcnmodel
import toolbox
import graphmapx

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ap = argparse.ArgumentParser()
ap.add_argument("--rgb", help="Use TFCN-RGB", action="store_true")
ap.add_argument("--grayscale", help="Use TFCN-G", action="store_true")
ap.add_argument("--extra", action="store_true", help="Run planner on extra images")
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

root = "."

if args.extra:
    ex = "extra"
else:
    ex = ""

W_path = os.path.join(
    root, "weights", WEIGHTS_DIR, "all" if args.outliers else "no outliers"
)
X_path = os.path.join(root, "dataset", ex, "X")
Y_path = os.path.join(root, "dataset", ex, "Y")
KP_path = os.path.join(root, "dataset", ex, "kp")
KN_path = os.path.join(root, "dataset", ex, "kn")

output_path__ = os.path.join(root, "results")
if not os.path.exists(output_path__):
    os.mkdir(output_path__)

X = list()
Y = list()
X_colored = list()

if args.extra:
    for imagename in sorted(os.listdir(X_path)):
        if imagename.endswith(".jpg"):
            x = toolbox.imread(os.path.join(X_path, imagename), color=COLOR)
            if args.rgb:
                x_colored = x
            else:
                x_colored = toolbox.imread(os.path.join(X_path, imagename), color="rgb")

            if args.grayscale:
                x = np.expand_dims(x, 2)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            dims = x.shape[1:]

            tfcn = tfcnmodel.TFCN(dims)
            tfcn.load(wp=os.path.join(W_path, "weights_%02d.hdf5" % (args.test)))

            r__ = 8

            time__ = time.time()

            y = tfcn.predict([x])
            y = np.squeeze(y[0])
            y = skimage.transform.resize(
                y, (y.shape[0] / r__, y.shape[1] / r__), anti_aliasing=True
            )

            y = (y - np.min(y)) / np.max(y)

            print("Time: %.3f s" % (time.time() - time__))

            if not os.path.exists(os.path.join(output_path__, "maps")):
                os.makedirs(os.path.join(output_path__, "maps"))

            cv.imwrite(
                os.path.join(
                    output_path__, "maps", "tfcn-output-%s.jpg" % (imagename[:-4])
                ),
                255 * y,
            )

            T = y

            keypoints_image = toolbox.imread(
                os.path.join(KP_path, imagename), color="rgb"
            )
            grid = toolbox.grid_list(np.squeeze(x), r__)
            keypoints = graphmapx.get_keypoints(keypoints_image, grid)

            C_file = os.path.join(W_path, "c_%02d.hdf5" % (args.test))
            c = np.loadtxt(C_file).item() / 2

            router = graphmapx.RouteEstimator(r=r__, c=c, grid=grid)
            G = router.tm2graph(T)

            output_path = os.path.join(
                output_path__,
                "paths/%s/%s/%s"
                % (COLOR, "all" if args.outliers else "no outliers", imagename[:-4]),
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for counter, (s, t) in enumerate(itertools.combinations(keypoints, 2)):
                path, found = router.route(G, s, t, T)

                font = cv.FONT_HERSHEY_SIMPLEX
                topLeftCornerOfText = (10, 40)
                fontScale = 1
                lineType = 2

                fontColor = (0, 255, 0)

                path_image = toolbox.draw_path(
                    x_colored, path, found=found, color=fontColor
                )

                toolbox.save_image(
                    os.path.join(output_path, "path-%d.jpg" % (counter + 1)),
                    [path_image],
                )

else:
    n_samples = 8 if args.outliers else 7

    for imagename in sorted(os.listdir(X_path)[:n_samples]):
        if imagename.endswith(".jpg"):
            x = toolbox.imread(os.path.join(X_path, imagename), color=COLOR)
            if args.rgb:
                x_colored = x
            else:
                x_colored = toolbox.imread(os.path.join(X_path, imagename), color="rgb")
            y = toolbox.imread(os.path.join(Y_path, imagename), color="grayscale")
            X.append(x)
            X_colored.append(x_colored)
            y = cv.resize(y, (125, 125)) / np.max(y)
            Y.append(y)

    X = np.array(X).astype(np.float32)
    Y = np.expand_dims(np.array(Y), 3).astype(np.float32)

    cross_val = sklearn.model_selection.LeaveOneOut()
    cross_val.get_n_splits(X)

    N = len(X)

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

        r = 8

        y = tfcn.predict(data.x_test)
        y = np.squeeze(y[0])
        y = skimage.transform.resize(y, (125, 125), anti_aliasing=True)

        y = (y - np.min(y)) / np.max(y)

        T = y
        GT = toolbox.imread(
            os.path.join(Y_path, "aerial%02d.jpg" % (test_index[0] + 1)), color="rgb"
        )

        keypoints_image = toolbox.imread(
            os.path.join(KP_path, "aerial%02d.jpg" % (test_index[0] + 1)), color="rgb"
        )
        grid = toolbox.grid_list(data.x_test[0], r)
        keypoints = graphmapx.get_keypoints(keypoints_image, grid)

        C_file = os.path.join(W_path, "c_%02d.hdf5" % (test_index + 1))
        c = np.loadtxt(C_file).item() / 2

        router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
        G = router.tm2graph(T)

        output_path = os.path.join(
            output_path__,
            "paths/%s/%s/%s"
            % (
                COLOR,
                "all" if args.outliers else "no outliers",
                "aerial%02d" % (test_index[0] + 1),
            ),
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        missions = list(itertools.combinations(keypoints, 2))

        for counter in tqdm.trange(len(missions), desc="  Map %d " % (test_index + 1)):
            (s, t) = missions[counter]

            path, found = router.route(G, s, t, T)

            score__ = toolbox.score(path, GT, r)

            font = cv.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (10, 40)
            fontScale = 1
            lineType = 2

            fontColor = (255, 0, 0) if score__ < 0.7 else (0, 255, 0)

            path_image = toolbox.draw_path(
                X_colored[test_index[0]], path, found=found, color=fontColor
            )
            cv.putText(
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
