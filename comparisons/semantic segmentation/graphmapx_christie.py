""" The graphmap module provides methods to generate 
    traversability graphs from traversability matrices.
    Tools for path computation are also be provided.
    Class RouteEstimator: defines a route estimator and its configuration.
    Method tdi2graph: builds a graph from a traversability matrix
    Method route: returns the best route between two regions
    Method draw_graph: saves a graph as an image (optionally, draws paths)
"""

import cv2
import numpy as np
import matplotlib
import networkx
import scipy.interpolate

def coord2(position, columns):
    """ Converts two-dimensional indexes to one-dimension coordinate
    """
    return position[0]*columns + position[1]

def get_keypoints(image, grid):
    """
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-image
    keypoints = detector.detect(reversemask)

    indexes = list()

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        size = int(keypoint.size)
        for i, (tlx, tly, sqsize) in enumerate(grid):
            if tlx <= x and x < tlx + sqsize:
                if tly <= y and y < tly + sqsize:
                    indexes.append(i)

    return indexes

def get_keypoints_yx(image, grid):
    """
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-image
    keypoints = detector.detect(reversemask)

    coords = list()

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        coords.append((y,x))

    return coords

def get_keypoints_overlap(image, grid, ov=3):
    """
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-image
    keypoints = detector.detect(reversemask)

    indexes = list()

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        size = int(keypoint.size)
        for i, (tlx, tly, sqsize) in enumerate(grid):
            if tlx <= x and x < tlx + sqsize/ov:
                if tly <= y and y < tly + sqsize/ov:
                    indexes.append(i)

    return indexes

# def draw_graph(G, filename="traversability-graph.png", path=[]):

#     G.vp.vfcolor = G.new_vertex_property("vector<double>")
#     G.ep.ecolor = G.new_edge_property("vector<double>")
#     G.ep.ewidth = G.new_edge_property("int")

#     for v in G.vertices():
#         diff = G.vp.traversability[v]
#         G.vp.vfcolor[v] = [1/(np.sqrt(diff)/100), 1/(np.sqrt(diff)/100), 1/(np.sqrt(diff)/100), 1.0]
#     for e in G.edges():
#         G.ep.ewidth[e] = 6
#         G.ep.ecolor[e] = [0.179, 0.203, 0.210, 0.8]
    
#     for i, v in enumerate(path):
#         G.vp.vfcolor[v] = [0, 0.640625, 0, 0.9]
#         if i < len(path) - 1:
#             for e in v.out_edges():
#                 if e.target() == path[i+1]:
#                     G.ep.ecolor[e] = [0, 0.640625, 0, 1]
#                     G.ep.ewidth[e] = 10

#     draw.graph_draw(G, pos=G.vp.pos, output_size=(1200, 1200), vertex_color=[0,0,0,1], vertex_fill_color=G.vp.vfcolor,\
#                     edge_color=G.ep.ecolor, edge_pen_width=G.ep.ewidth, output=filename, edge_marker_size=4)

# Ramer-Douglas-Peucker from https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line

def _vec2d_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _vec2d_sub(p1, p2):
    return (p1[0]-p2[0], p1[1]-p2[1])

def _vec2d_mult(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

def ramerdouglas(line, dist):
    """Does Ramer-Douglas-Peucker simplification of a curve with 'dist' threshold.

    'line' is a list-of-tuples, where each tuple is a 2D coordinate

    Usage is like so:

    >>> myline = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
    >>> simplified = ramerdouglas(myline, dist = 1.0)
    """

    if len(line) < 3:
        return line

    (begin, end) = (line[0], line[-1]) if line[0] != line[-1] else (line[0], line[-2])

    distSq = []
    for curr in line[1:-1]:
        tmp = (
            _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin),
            _vec2d_sub(curr, begin)) ** 2 / _vec2d_dist(begin, end))
        distSq.append(tmp)

    maxdist = max(distSq)
    if maxdist < dist ** 2:
        return [begin, end]

    pos = distSq.index(maxdist)
    return (ramerdouglas(line[:pos + 2], dist) + 
            ramerdouglas(line[pos + 1:], dist)[1:])

def g0(G, v, u):
    return (v, u, {'weight': 1})

def g(G, v, u):
    w = np.array([5, 2, 5])
    phi1 = 0
    phi2 = 0
    phi3 = 0
    # Compute phi1
    distance_to_closest_non_road = None
    # perform breadth search
    for _, n in networkx.bfs_edges(G, v):
        if G.nodes[n]['class_'] != 1:
            (y1, x1) = G.nodes[v]['pos'][1], G.nodes[v]['pos'][0]
            (y2, x2) = G.nodes[n]['pos'][1], G.nodes[n]['pos'][0]
            distance_to_closest_non_road = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            break
    phi1 = 1.0/distance_to_closest_non_road
    # Compute phi2
    if G.nodes[u]['class_'] != 1: phi2 = 1.0
    else: phi2 = 0.0
    # Compute phi3
    if G.nodes[v]['class_'] != 1 and G.nodes[v]['class_'] != 3: phi3 = float('inf')
    else:
        distance_to_closest_non_road_or_grass = None
        # perform breadth search
        for _, n in networkx.bfs_edges(G, v):
            if G.nodes[n]['class_'] != 1 and G.nodes[n]['class_'] != 3:
                (y1, x1) = G.nodes[v]['pos'][1], G.nodes[v]['pos'][0]
                (y2, x2) = G.nodes[n]['pos'][1], G.nodes[n]['pos'][0]
                distance_to_closest_non_road_or_grass = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                break
        phi3 = 1.0/distance_to_closest_non_road_or_grass
    phi = np.array([phi1, phi2, phi3])
    return np.dot(w, phi)


class RouteEstimator:
    """
    """
    def __init__(self, r, grid):
        self.r = r
        self.grid = grid

    def seg2graph(self, segmentation):

        G = networkx.DiGraph()

        for i, row in enumerate(segmentation):
            for j, element in enumerate(row):
                index = coord2((i,j), segmentation.shape[1])
                G.add_node( index,
                            pos=[j,i],
                            class_=segmentation[i][j],
                            # cut=True if segmentation[i][j] in [0, 2, 4, 5] else False
                            cut=False
                )

        edges = list()

        for v in [vv for vv in G.nodes() if not G.nodes[vv]['cut']]:

            (i, j) = G.nodes[v]['pos'][1], G.nodes[v]['pos'][0]

            top, bottom, left, right = (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            if i-1 > -1:
                u = coord2(top, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if i+1 < segmentation.shape[0]:
                u = coord2(bottom, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if j-1 > -1:
                u = coord2(left, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if j+1 < segmentation.shape[1]:
                u = coord2(right, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))

            topleft, topright, bottomleft, bottomright = (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
            if i-1 > -1 and j-1 > -1:
                u = coord2(topleft, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if i-1 > -1 and j+1 < segmentation.shape[1]:
                u = coord2(topright, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if i+1 < segmentation.shape[0] and j-1 > -1:
                u = coord2(bottomleft, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))
            if i+1 < segmentation.shape[0] and j+1 < segmentation.shape[1]:
                u = coord2(bottomright, segmentation.shape[1])
                if not G.nodes[u]['cut']:
                    edges.append(g0(G, v, u))

        G.add_edges_from(edges)

        for v, u, d in G.edges(data=True):
            d['weight'] = g(G, v, u)

        del edges

        return G

    # def map_from_source(self, G, source):
    #     dist, pred = search.dijkstra_search(G, G.ep.weight, source, Visitor())
    #     return dist, pred

    # def route(self, G, source, target):
    #     try:
    #         path = networkx.shortest_path(G, source, target, 'weight')
    #         found = True
    #     except networkx.exception.NetworkXNoPath:
    #         path = [source, target]
    #         found = False
    #     return path, found    

    def route(self, G, source, target, segmentation):
        def dist(a, b):
            (y1, x1) = G.nodes[a]['pos'][1], G.nodes[a]['pos'][0]
            (y2, x2) = G.nodes[b]['pos'][1], G.nodes[b]['pos'][0]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        def dist2(a, b):
            (y1, x1) = G.nodes[a]['pos'][1], G.nodes[a]['pos'][0]
            (y2, x2) = G.nodes[b]['pos'][1], G.nodes[b]['pos'][0]
            xs, xe, ys, ye = 0, 0, 0, 0
            if x1 < x2: xs, xe = x1, x2
            else: xs, xe = x2, x1
            if y1 < y2: ys, ye = y1, y2
            else: ys, ye = y2, y1
            return np.mean(segmentation[ys:ye, xs:xe])
        try:
            path = networkx.astar_path(G, source, target, dist, 'weight')
            centers = list()
            for k in path:
                if G.nodes[k]['class_'] != 1 and G.nodes[k]['class_'] != 3:
                    path = [source, target]
                    centers = list()
                    for k in path:
                        tly, tlx, size = self.grid[k]
                        centers.append((int(tlx+(size/2)), int(tly+(size/2))))
                    path = centers
                    found = False
                    return path, found
                tly, tlx, size = self.grid[k]
                centers.append((int(tlx+(size/2)), int(tly+(size/2))))
            # # Ramer-Douglas-Peucker from https://stackoverflow.com/questions/2573997/reduce-number-of-points-in-line
            # centers = ramerdouglas(centers, self.r*1.5)
            # Linear interpolation from https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            X = np.array(centers)
            alpha = np.linspace(0, 1, len(path))
            distance = np.cumsum(np.sqrt(np.sum(np.diff(X, axis=0)**2, axis=1)))
            distance = np.insert(distance, 0, 0)/distance[-1]
            interpolator =  scipy.interpolate.interp1d(distance, X, kind='slinear', axis=0)
            curve = interpolator(alpha)
            curve = np.round(curve).astype(int)
            # Spline smoothing from https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            X = np.array(curve)
            distance = np.cumsum(np.sqrt(np.sum(np.diff(X, axis=0)**2, axis=1)))
            distance = np.insert(distance, 0, 0)/distance[-1]
            splines = [scipy.interpolate.UnivariateSpline(distance, coords, k=2) for coords in X.T]
            points_fitted = np.vstack([spl(alpha) for spl in splines]).T
            points_fitted = np.round(points_fitted).astype(int)
            centers = points_fitted
            # TODO: FIX WORKAROUND FOR IMAGES 1000x1000
            # centers = np.clip(centers, 0, 999)
            # Returning pixel coordinates
            path = centers
            found = True
        except (networkx.exception.NetworkXNoPath, ValueError):
            path = [source, target]
            centers = list()
            for k in path:
                tly, tlx, size = self.grid[k]
                centers.append((int(tlx+(size/2)), int(tly+(size/2))))
            path = centers
            found = False
        return path, found
