import os
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument(
    "--use", type=int, default=7, required=False, help="Select how many images to use"
)
args = ap.parse_args()

output_path = "output/"
output_file = "data-%d.json" % (args.use)


def extract_features(region):
    features = []
    # 1) The average value r of the red content in the image
    features.append(np.mean(region[:, :, 0]))
    # 2) The average value g of the green content in the image.
    features.append(np.mean(region[:, :, 1]))
    # 3) The average value b of the blue content in the image.
    features.append(np.mean(region[:, :, 2]))
    # 4) The mean m of the gray image.
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mu = np.mean(gray_region)
    features.append(mu)
    # 5) The standard deviation σ of the gray image.
    features.append(np.array(np.std(gray_region)))
    # 6) The smoothness R of the gray image.
    H = np.histogram(gray_region, bins=256)[0] + 1
    total = np.sum(H)
    numerator = np.sum([(zi / total) * (zi - mu) ** 2 for zi in H])
    R = numerator / (1 + numerator)
    features.append(R)
    # 7) The third moment μ3 . The feature is a measurement of the skewness of a histogram.
    features.append(np.sum([(zi / total) * (zi - mu) ** 3 for zi in H]))
    # 8) The uniformity U.
    features.append(np.sum([(zi / total) ** 2 for zi in H]))
    # 9) The entropy e. The feature is a measurement of randomness for the all gray levels of the intensity histogram.
    e = -np.sum([(zi / total) * np.log2(zi / total) for zi in H])
    features.append(e)

    return features


MSE = []


def main_experiment():
    """ """
    import time
    import json
    import tqdm
    import itertools
    import scipy.interpolate
    import pickle

    import graphmapx

    import sys

    sys.path.append("../../")
    import toolbox

    resolution = 15

    dataset_path = "../../dataset/X/"
    ground_truth_path = "../../dataset/Y/"
    positive_keypoints_path = "../../dataset/kp/"
    negative_keypoints_path = "../../dataset/kn/"
    w_path = os.path.join("svm")

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

        filename = os.path.join(w_path, "svm-%d-%02d.pkl" % (args.use, i + 1))
        clf = pickle.load(open(filename, "rb"))

        start_matrix_time = time.time()
        y = np.zeros((1000, 1000))
        x = image
        grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
        for g in grid:
            [tlx, tly, r] = g
            region = x[tly : tly + r, tlx : tlx + r]
            if region.shape != (resolution, resolution, 3):
                continue
            features = extract_features(region)
            traversability = clf.predict([features])[0]
            y[tly : tly + r, tlx : tlx + r] = (
                y[tly : tly + r, tlx : tlx + r] + traversability
            ) / 2

        y = (y - np.min(y)) / np.max(y)
        y = cv2.resize(y, (125, 125))
        y = y.astype(np.float32)

        t_matrix = y
        matrix_time = time.time() - start_matrix_time

        gt = toolbox.imread(os.path.join(ground_truth_path, image_path))
        gt_small = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt_small = cv2.resize(gt_small, (125, 125))
        gt_small = (gt_small - np.min(gt_small)) / np.max(gt_small)

        MSE.append((((y - gt_small) ** 2).mean(axis=None)))

        r = 8
        grid = toolbox.grid_list(image, r)

        C_file = os.path.join(w_path, "c_%d_%02d.hdf5" % (args.use, i + 1))
        c = 1.25 * np.loadtxt(C_file).item()

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
            results["traversability_function"] = "TFCN"
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
            results["traversability_function"] = "TFCN"
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

if len(MSE) > 0:
    for mse in MSE:
        print("%.3f" % mse)
