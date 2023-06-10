import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

ap = argparse.ArgumentParser()
args = ap.parse_args()

try:
    from google.colab import drive, auth

    drive.mount("/content/drive")
    root = "drive/My Drive/Colab Notebooks/TFCN"
except:
    root = "."

output_path = os.path.join(root, "results")
output_file = "data-peterson.json"

rgb_classes = [
    [0, 0, 0],  # 0 Unknown
    [255, 0, 0],  # 1 Road
    [0, 0, 255],  # 2 Building/Containers
    [0, 204, 0],  # 3 Grass
    [0, 77, 0],  # 4 Trees/Bushes/Vegetation
    [0, 0, 77],  # 5 Vehicle
]
rgb_classes = np.array(rgb_classes)


def rgb_to_classes(rgb_label):
    h, w, d = rgb_label.shape
    class_label = np.zeros((h, w, 1))
    for y in range(h):
        for x in range(w):
            pixel = rgb_label[y][x]
            class_ = np.argmin(np.linalg.norm(pixel - rgb_classes, axis=1))
            class_label[y][x] = class_
    return class_label


def classes_to_rgb(class_label):
    h, w = list(class_label.shape)
    rgb_label = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            class_ = class_label[y, x].item()
            pixel = rgb_classes[class_]
            rgb_label[y][x] = pixel
    return rgb_label


def show_images(images, block=True):
    def plot():
        n = len(images)
        if n == 1:
            fig, (ax0) = plt.subplots(ncols=1)
            ax0.imshow(images[0], cmap="gray", interpolation="bicubic")
            ax0.axes.get_xaxis().set_ticks([])
            ax0.axes.get_yaxis().set_visible(False)
        else:
            fig, axes = plt.subplots(ncols=n, figsize=(4 * n, 4))
            for ax, image in zip(axes, images):
                ax.imshow(image, cmap="gray", interpolation="bicubic")
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.show()

    if block:
        plot()
    else:
        multiprocessing.Process(target=plot).start()


segmentation_time = (
    np.array([15148340, 144194785, 41917580, 3025772, 5548635, 3843030, 15958191])
    * 10**-6
)


def main_experiment():
    """ """
    import cv2
    import time
    import json
    import tqdm
    import itertools
    import scipy.interpolate

    import torch
    import torchvision

    import graphmapx_peterson as graphmapx

    import sys

    sys.path.append("../../")

    import toolbox

    dataset_path = os.path.join(root, "../../dataset/X/")
    ground_truth_path = os.path.join(root, "../../dataset/Y/")
    positive_keypoints_path = os.path.join(root, "../../dataset/kp/")
    negative_keypoints_path = os.path.join(root, "../../dataset/kn/")
    w_path = os.path.join(root, "weights")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_samples = 7
    images = ["aerial%02d.jpg" % (i + 1) for i in range(n_samples)]

    dataset = list()
    for (_, _, filenames) in os.walk(ground_truth_path):
        dataset.extend(filenames)
        break

    selected = list(set(dataset).intersection(images)) if len(images) > 0 else dataset
    selected.sort()

    data = list()

    device = torch.device("cpu")
    print("Using", str(device).upper())

    for i in tqdm.trange(len(selected), desc="Input image "):
        # for i in range(len(selected)):

        segmentation_path = os.path.join(
            root, "results_ale", "Result%d" % (i + 1), "Crf"
        )

        r = 8

        image_path = selected[i][:-4] + ".jpg"

        positive_keypoints = toolbox.imread(
            os.path.join(positive_keypoints_path, image_path)
        )
        negative_keypoints = toolbox.imread(
            os.path.join(negative_keypoints_path, image_path)
        )

        segmentation = toolbox.imread(
            os.path.join(segmentation_path, selected[i][:-4] + ".png"), color="rgb"
        )

        matrix_time = time.time()
        segmentation = rgb_to_classes(segmentation)
        segmentation = cv2.resize(segmentation, (125, 125))

        costs = segmentation.copy()
        costs[costs == 0] = float("inf")
        costs[costs == 1] = 16
        costs[costs == 2] = float("inf")
        costs[costs == 3] = 27
        costs[costs == 4] = float("inf")
        costs[costs == 5] = float("inf")

        t_matrix = cv2.resize(costs, (125, 125))

        # DILATION STEP NEEDED

        matrix_time = time.time() - matrix_time + segmentation_time[i]

        gt = toolbox.imread(os.path.join(ground_truth_path, image_path))
        grid = toolbox.grid_list(gt, r)

        c = 30

        router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)

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
            results["traversability_function"] = "Semantic Segmentation"
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (10, 40)
            fontScale = 1
            lineType = 2

            fontColor = (255, 0, 0) if score < 0.7 else (0, 255, 0)

            img = toolbox.imread(
                os.path.join(dataset_path, selected[i][:-4] + ".jpg"), color="rgb"
            )
            path_image = toolbox.draw_path(img, path, found=found, color=fontColor)
            cv2.putText(
                path_image,
                "Score: %.2f" % (score),
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )

            p_path = os.path.join(
                output_path, "paths", "peterson", "aerial%02d" % (i + 1), "positive"
            )
            if not os.path.exists(p_path):
                os.makedirs(p_path)
            toolbox.save_image(
                os.path.join(p_path, "path-%d.jpg" % (counter + 1)), [path_image]
            )

        keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
        combinations = list(itertools.combinations(keypoints, 2))

        for counter in range(len(combinations)):

            (s, t) = combinations[counter]

            start_route_time = time.time()
            path, found = router.route(G, s, t, t_matrix)
            route_time = time.time() - start_route_time

            results = dict()
            results["image"] = image_path
            results["traversability_function"] = "Semantic Segmentation"
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            topLeftCornerOfText = (10, 40)
            fontScale = 1
            lineType = 2

            fontColor = (255, 0, 0) if score < 0.7 else (0, 255, 0)

            img = toolbox.imread(
                os.path.join(dataset_path, selected[i][:-4] + ".jpg"), color="rgb"
            )
            path_image = toolbox.draw_path(img, path, found=found, color=fontColor)
            cv2.putText(
                path_image,
                "Score: %.2f" % (score),
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )

            p_path = os.path.join(
                output_path, "paths", "peterson", "aerial%02d" % (i + 1), "negative"
            )
            if not os.path.exists(p_path):
                os.makedirs(p_path)
            toolbox.save_image(
                os.path.join(p_path, "path-%d.jpg" % (counter + 1)), [path_image]
            )

        with open(os.path.join(output_path, output_file), "w") as datafile:
            json.dump(data, datafile, indent=4)


def get_results(datapath):
    """ """
    import json
    import numpy

    with open(datapath) as datafile:
        data = json.load(datafile)

    print("Total samples:", len(data))

    n_samples = 7
    images = ["aerial%02d.jpg" % (i + 1) for i in range(n_samples)]

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
    print("Avg path safety: %.3f (+/- %.3f)" % (numpy.mean(score), numpy.std(score)))

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
