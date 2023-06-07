import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "--use", type=int, default=7, required=False, help="Select how many images to use"
)
args = ap.parse_args()

output_path = "output/"
output_file = "data-%d.json" % (args.use)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main_experiment():
    """ """
    import cv2
    import time
    import json
    import tqdm
    import numpy as np
    import itertools
    import scipy.interpolate

    import sys

    sys.path.append("../../")
    import toolbox
    import graphmapx

    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    resolution = 15
    if args.use == 8:
        classes = ["road", "building", "grass", "vegetation", "vehicle", "sand"]
        cost_vector = [-100, 200, -60, 200, 150, -20]
    elif args.use == 7:
        classes = ["road", "building", "grass", "vegetation", "vehicle"]
        cost_vector = [-100, 200, -60, 200, 150]

    dataset_path = "../../dataset/X/"
    ground_truth_path = "../../dataset/Y/"
    positive_keypoints_path = "../../dataset/kp/"
    negative_keypoints_path = "../../dataset/kn/"
    w_path = os.path.join("weights%d" % (args.use))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = ["aerial%02d.jpg" % (i + 1) for i in range(args.use)]

    dataset = list()
    for (_, _, filenames) in os.walk(ground_truth_path):
        dataset.extend(filenames)
        break

    selected = list(set(dataset).intersection(images)) if len(images) > 0 else dataset
    selected.sort()

    data = list()

    for i in tqdm.trange(len(selected), desc="Input image "):
        # for i in range(len(selected)):

        image_path = selected[i]
        image = toolbox.imread(os.path.join(dataset_path, image_path), color="rgb")
        positive_keypoints = toolbox.imread(
            os.path.join(positive_keypoints_path, image_path)
        )
        negative_keypoints = toolbox.imread(
            os.path.join(negative_keypoints_path, image_path)
        )

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
            model.add(
                Conv2D(filters=50, kernel_size=5, padding="same", activation="tanh")
            )
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
        weights_path = os.path.join(w_path, "weights_%02d.hdf5" % (i + 1))

        # Carregamento da melhor combinação de pesos obtida durante o treinamento
        CNN.load_weights(weights_path)

        thresholds = np.loadtxt(
            os.path.join(w_path, "thresholds-%d.txt" % (i + 1)), dtype=float
        )

        start_matrix_time = time.time()
        # Teste
        y_img = np.zeros((1000, 1000))
        x = toolbox.imread(
            os.path.join(dataset_path, "aerial%02d.jpg" % (i + 1)), color="rgb"
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
        t_matrix = T.astype(np.float32)
        matrix_time = time.time() - start_matrix_time

        gt = toolbox.imread(os.path.join(ground_truth_path, image_path))

        r = 8
        grid = toolbox.grid_list(image, r)

        c = c_unknown + 50

        router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)
        G = router.tm2graph(t_matrix)

        start_graph_time = time.time()
        G = router.tm2graph(t_matrix)
        graph_time = time.time() - start_graph_time

        keypoints = graphmapx.get_keypoints(positive_keypoints, grid)
        combinations = list(itertools.combinations(keypoints, 2))

        for counter in range(len(combinations)):

            (s, t) = combinations[counter]

            start_route_time = time.time()
            path, found = router.route(G, s, t, t_matrix)
            route_time = time.time() - start_route_time

            score = toolbox.score(path, gt, r)

            results = dict()
            results["image"] = image_path
            results["traversability_function"] = "CNN Region Classification"
            results["region_size"] = r
            results["cut_threshold"] = c
            results["path_existence"] = True
            results["matrix_build_time"] = matrix_time
            results["graph_build_time"] = graph_time
            results["path_build_time"] = route_time
            results["path_found"] = found
            results["path_score"] = score if found else 0.0
            results["path_coordinates"] = [(int(p[0]), int(p[1])) for p in path]
            results["path_length"] = len(path)

            data.append(results)

        keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
        combinations = list(itertools.combinations(keypoints, 2))

        for counter in range(len(combinations)):

            (s, t) = combinations[counter]

            start_route_time = time.time()
            path, found = router.route(G, s, t, t_matrix)
            route_time = time.time() - start_route_time

            results = dict()
            results["image"] = image_path
            results["traversability_function"] = "CNN Region Classification"
            results["region_size"] = r
            results["cut_threshold"] = c
            results["matrix_build_time"] = matrix_time
            results["graph_build_time"] = graph_time
            results["path_build_time"] = route_time
            results["path_existence"] = False
            results["path_found"] = found
            results["path_score"] = 1.0 if not found else 0.0
            results["path_coordinates"] = [(int(p[0]), int(p[1])) for p in path]
            results["path_length"] = len(path)

            data.append(results)

        with open(os.path.join(output_path, output_file), "w") as datafile:
            json.dump(data, datafile, indent=4)


def get_results(datapath):
    """ """
    import json
    import numpy

    with open(datapath) as datafile:
        data = json.load(datafile)

    print("Total samples:", len(data))

    images = ["aerial%02d.jpg" % (i + 1) for i in range(args.use)]

    matrix_time = list()
    graph_time = list()
    route_time = list()
    score = list()
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    lengths = list()

    for sample in data:
        if sample["image"] in images:

            matrix_time.append(sample["matrix_build_time"])
            graph_time.append(sample["graph_build_time"])
            route_time.append(sample["path_build_time"])

            score.append(sample["path_score"])
            if (
                sample["path_existence"]
                and sample["path_found"]
                and sample["path_score"] > 0.7
            ):
                lengths.append(sample["path_length"])

            if sample["path_existence"] and sample["path_found"]:
                true_positive += 1
            elif not sample["path_existence"] and not sample["path_found"]:
                true_negative += 1
            elif not sample["path_existence"] and sample["path_found"]:
                false_positive += 1
            elif sample["path_existence"] and not sample["path_found"]:
                false_negative += 1

    matrix_time = numpy.array(matrix_time)
    graph_time = numpy.array(graph_time)
    route_time = numpy.array(route_time)

    print(
        "Avg matrix build time: %.3f (+/- %.3f)"
        % (numpy.mean(matrix_time), numpy.std(matrix_time))
    )
    print(
        "Avg graph build time: %.3f (+/- %.3f)"
        % (numpy.mean(graph_time), numpy.std(graph_time))
    )
    print(
        "Avg mapping time: %.3f (+/- %.3f)"
        % (numpy.mean(matrix_time + graph_time), numpy.std(matrix_time + graph_time))
    )
    print(
        "Avg route build time: %.3f (+/- %.3f)"
        % (numpy.mean(route_time), numpy.std(route_time))
    )
    print(
        "Avg total time: %.3f (+/- %.3f)"
        % (
            numpy.mean(matrix_time + graph_time + route_time),
            numpy.std(matrix_time + graph_time + route_time),
        )
    )

    print("Avg path length: %d" % int(numpy.round(numpy.mean(lengths))))
    print("Avg path quality: %.3f (+/- %.3f)" % (numpy.mean(score), numpy.std(score)))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )

    frdr = recall
    irdr = true_negative / (true_negative + false_positive)
    fdr = accuracy

    print("Accuracy: %.3f" % accuracy)
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)

    print("Feasible route detection rate: %.3f" % frdr)
    print("Infeasible route detection rate: %.3f" % irdr)
    print("Feasiblity detection rate: %.3f" % fdr)

    print("Evaluated samples:", len(matrix_time))


if not os.path.exists(os.path.join(output_path, output_file)):
    main_experiment()
get_results(os.path.join(output_path, output_file))
