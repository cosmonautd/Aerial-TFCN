import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--rgb", help="Use TFCN-RGB", action="store_true")
ap.add_argument("--grayscale", help="Use TFCN-G", action="store_true")
ap.add_argument("--outliers", help="Include outliers", action="store_true")
ap.add_argument("--no-normalization", help="Do not normalize the traversability matrix", action="store_true", default=False)
ap.add_argument("--fixed-cut-threshold", help="Use a fixed cut threshold for region cutting", type=float, default=None)
ap.add_argument("--disable-path-smoothing", help="Disable path smoothing", action="store_true", default=False)
args = ap.parse_args()

try:
    from google.colab import drive, auth
    drive.mount('/content/drive')
    root = 'drive/My Drive/Colab Notebooks/TFCN + Conv2DTranspose (keras)'
except:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = '.'

if args.rgb: COLOR = 'rgb'
elif args.grayscale: COLOR = 'grayscale'

output_path = os.path.join(root, 'results')
output_file = 'data-%s-%s.json' % ('all' if args.outliers else 'no-outliers', COLOR)

def main_experiment():
    """
    """
    import cv2
    import time
    import json
    import tqdm
    import numpy as np
    import itertools
    import scipy.interpolate
    import skimage.transform

    import toolbox
    import tfcnmodel
    import graphmapx

    dataset_path = os.path.join(root, 'dataset/X/')
    ground_truth_path = os.path.join(root, 'dataset/Y/')
    positive_keypoints_path = os.path.join(root, 'dataset/kp/')
    negative_keypoints_path = os.path.join(root, 'dataset/kn/')
    w_path = os.path.join(root, 'weights', COLOR, 'all' if args.outliers else 'no outliers')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_samples = 8 if args.outliers else 7
    images = ['aerial%02d.jpg' % (i+1) for i in range(n_samples)]

    dataset = list()
    for (_, _, filenames) in os.walk(ground_truth_path):
        dataset.extend(filenames)
        break

    selected = list(set(dataset).intersection(images)) if len(images) > 0 else dataset
    selected.sort()

    data = list()

    for i in tqdm.trange(len(selected), desc=" Input image "):
    # for i in range(len(selected)):

        image_path = selected[i]
        image = toolbox.imread(os.path.join(dataset_path, image_path), color=COLOR)
        if COLOR == 'grayscale': image = np.expand_dims(image, 2)
        positive_keypoints = toolbox.imread(os.path.join(positive_keypoints_path, image_path))
        negative_keypoints = toolbox.imread(os.path.join(negative_keypoints_path, image_path))

        r = 8

        dims = image.shape
        tfcn = tfcnmodel.TFCN(dims)
        tfcn.load(wp=os.path.join(w_path, 'weights_%02d.hdf5' % (i+1)))

        input__ = np.expand_dims(image, axis=0).astype(np.float32)

        start_matrix_time = time.time()
        t_matrix = tfcn.predict(input__)[0]
        t_matrix = np.squeeze(t_matrix)
        t_matrix = skimage.transform.resize(t_matrix, (125, 125), anti_aliasing=True)

        if not args.no_normalization:
            t_matrix = (t_matrix - np.min(t_matrix))/np.max(t_matrix)

        matrix_time = time.time() - start_matrix_time

        gt = toolbox.imread(os.path.join(ground_truth_path, image_path))

        grid = toolbox.grid_list(image, r)

        # th = cv2.calcHist(t_matrix, [0], None, [100], [0, 1])
        # th = th.flatten()
        # tv = np.arange(0, 1, 1/100)
        # c = np.sum(th*tv)/np.sum(th)

        if args.fixed_cut_threshold is not None:
            c = args.fixed_cut_threshold
        else:
            C_file = os.path.join(w_path, 'c_%02d.hdf5' % (i + 1))
            c = np.loadtxt(C_file).item()/2

        router = graphmapx.RouteEstimator(r=r, c=c, grid=grid)

        start_graph_time = time.time()
        G = router.tm2graph(t_matrix)
        graph_time = time.time() - start_graph_time

        keypoints = graphmapx.get_keypoints(positive_keypoints, grid)
        combinations = list(itertools.combinations(keypoints, 2))

        smooth_paths = not args.disable_path_smoothing

        for counter in range(len(combinations)):

            (s, t) = combinations[counter]

            start_route_time = time.time()
            path, found = router.route(G, s, t, t_matrix, {"smooth": smooth_paths})
            route_time = time.time() - start_route_time

            score = toolbox.score(path, gt, r)

            results = dict()
            results['image'] = image_path
            results['traversability_function'] = 'TFCN'
            results['region_size'] = r
            results['cut_threshold'] = c
            results['path_existence'] = True
            results['matrix_build_time'] = matrix_time
            results['graph_build_time'] = graph_time
            results['path_build_time'] = route_time
            results['path_found'] = found
            results['path_score'] = score if found else 0.0
            results['path_coordinates'] = [(int(p[0]), int(p[1])) for p in path]
            results['path_length'] = len(path)

            data.append(results)

        keypoints = graphmapx.get_keypoints(negative_keypoints, grid)
        combinations = list(itertools.combinations(keypoints, 2))

        for counter in range(len(combinations)):

            (s, t) = combinations[counter]

            start_route_time = time.time()
            path, found = router.route(G, s, t, t_matrix, {"smooth": smooth_paths})
            route_time = time.time() - start_route_time

            results = dict()
            results['image'] = image_path
            results['traversability_function'] = 'TFCN'
            results['region_size'] = r
            results['cut_threshold'] = c
            results['matrix_build_time'] = matrix_time
            results['graph_build_time'] = graph_time
            results['path_build_time'] = route_time
            results['path_existence'] = False
            results['path_found'] = found
            results['path_score'] = 1.0 if not found else 0.0
            results['path_coordinates'] = [(int(p[0]), int(p[1])) for p in path]
            results['path_length'] = len(path)

            data.append(results)

        with open(os.path.join(output_path, output_file), 'w') as datafile:
            json.dump(data, datafile, indent=4)

def get_results(datapath):
    """
    """
    import json
    import numpy

    with open(datapath) as datafile:
        data = json.load(datafile)

    n_samples = 8 if args.outliers else 7
    images = ['aerial%02d.jpg' % (i+1) for i in range(n_samples)]

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
        if sample['image'] in images:

            matrix_time.append(sample['matrix_build_time'])
            graph_time.append(sample['graph_build_time'])
            route_time.append(sample['path_build_time'])

            score.append(sample['path_score'])
            if sample['path_existence'] and sample['path_found'] and sample['path_score'] > 0.7:
                lengths.append(sample['path_length'])
            
            if sample['path_existence'] and sample['path_found']:
                true_positive += 1
            elif not sample['path_existence'] and not sample['path_found']:
                true_negative += 1
            elif not sample['path_existence'] and sample['path_found']:
                false_positive += 1
            elif sample['path_existence'] and not sample['path_found']:
                false_negative += 1
    
    matrix_time = numpy.array(matrix_time)
    graph_time = numpy.array(graph_time)
    route_time = numpy.array(route_time)

    # print("Avg path length: %d" % int(numpy.round(numpy.mean(lengths))))
    print("Avg path quality: %.3f (+/- %.3f)" % (numpy.mean(score), numpy.std(score)))

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)

    frdr = recall
    irdr = true_negative/(true_negative + false_positive)
    fdr  = accuracy

    # print("Accuracy: %.3f" % accuracy)
    # print("Precision: %.3f" % precision)
    # print("Recall: %.3f" % recall)

    print("Feasible route detection rate: %.3f" % frdr)
    print("Infeasible route detection rate: %.3f" % irdr)
    print("Feasiblity detection rate: %.3f" % fdr)

    print("Avg matrix build time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time), numpy.std(matrix_time)))
    print("Avg graph build time: %.3f (+/- %.3f)" % (numpy.mean(graph_time), numpy.std(graph_time)))
    print("Avg mapping time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time + graph_time), numpy.std(matrix_time + graph_time)))
    print("Avg route build time: %.3f (+/- %.3f)" % (numpy.mean(route_time), numpy.std(route_time)))
    print("Avg total time: %.3f (+/- %.3f)" % (numpy.mean(matrix_time + graph_time + route_time), numpy.std(matrix_time + graph_time + route_time)))

    print("Evaluated samples:", len(matrix_time))

# if not os.path.exists(os.path.join(output_path, output_file)):
main_experiment()
get_results(os.path.join(output_path, output_file))