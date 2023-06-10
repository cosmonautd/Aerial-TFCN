import os
import multiprocessing
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"


def imread(path, color="rgb"):
    """Loads image from path"""
    if color == "grayscale":
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
    elif color == "rgb":
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    elif color == "hsv":
        return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2HSV)
    elif color == "rgbh":
        img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        h, w = img.shape[0:2]
        R = img[:, :, 0].flatten()
        G = img[:, :, 1].flatten()
        B = img[:, :, 2].flatten()
        c1 = np.arctan2(R, np.maximum(G, B)).reshape((h, w))
        c2 = np.arctan2(G, np.maximum(R, B)).reshape((h, w))
        c3 = np.arctan2(B, np.maximum(G, R)).reshape((h, w))
        H = np.arctan2(np.cbrt(G - B), 2 * R - G - B).reshape((h, w))
        return np.dstack((c1, c2, c3, H))


def grid_list(image, r):
    """Returns a list of square coordinates representing a grid over image
    Every square has length and height equals to r
    """
    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    # assertions that guarantee the square grid contains all pixels
    assert r > 0, "Parameter r must be larger than zero"
    if (height / r).is_integer() and (width / r).is_integer():
        glist = []
        for toplefty in range(0, height, r):
            for topleftx in range(0, width, r):
                glist.append((topleftx, toplefty, r))
        return glist
    else:
        new_height = int(r * np.floor(height / r))
        new_width = int(r * np.floor(width / r))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height) / 2)
            x_edge = int((width - new_width) / 2)
            glist = []
            for toplefty in range(y_edge, y_edge + new_height, r):
                for topleftx in range(x_edge, x_edge + new_width, r):
                    glist.append((topleftx, toplefty, r))
            return glist
        else:
            raise ValueError("r probably larger than image dimensions")


def grid_list_overlap(image, r, overlap=0.5):
    """Returns a list of square coordinates representing a grid over image with overlapping
    Every square has length and height equals to r
    """
    height, width, _ = image.shape
    ov = int(1 / overlap)
    # assertions that guarantee the square grid contains all pixels
    assert r > 0, "Parameter r must be larger than zero"
    if (height / r).is_integer() and (width / r).is_integer():
        glist = []
        for toplefty in range(0, ov * height, r):
            for topleftx in range(0, ov * width, r):
                glist.append((int(topleftx / ov), int(toplefty / ov), r))
        return glist
    else:
        new_height = int(r * np.floor(height / r))
        new_width = int(r * np.floor(width / r))
        if new_height > 0 and new_width > 0:
            y_edge = int((height - new_height) / 2)
            x_edge = int((width - new_width) / 2)
            glist = []
            for toplefty in range(y_edge, (y_edge + ov * new_height), r):
                for topleftx in range(x_edge, (x_edge + ov * new_width), r):
                    glist.append((int(topleftx / ov), int(toplefty / ov), r))
            return glist
        else:
            raise ValueError("r probably larger than image dimensions")


def score(path, ground_truth, r):
    score_ = 1.0
    penalty = 0.03
    T = list()
    for px in path:
        h, w, _ = ground_truth.shape
        a = max(0, px[0] - int(r / 2))
        b = min(h - 1, px[0] + int(r / 2))
        c = max(0, px[1] - int(r / 2))
        d = min(w - 1, px[1] + int(r / 2))
        t = ground_truth[a:b, c:d]
        t = t.mean(axis=2) / 255
        t = cv.erode(t, np.ones((int(r / 2), int(r / 2)), np.uint8), iterations=1)
        T.append(np.mean(t))
    for i, t in enumerate(T):
        if i < len(T) - 1 and i > 0:
            if t < 0.5:
                score_ = np.maximum(0, score_ - penalty * (1 - t))
    return score_


def draw_path(image, path, color=(0, 255, 0), found=False):
    image_copy = image.copy()
    if len(image_copy.shape) < 3:
        image_copy = cv.cvtColor(image_copy, cv.COLOR_GRAY2RGB)
    centers = [(p[0], p[1]) for p in path]
    cv.circle(image_copy, centers[0][::-1], 6, color, -1)
    cv.circle(image_copy, centers[-1][::-1], 12, color, -1)
    if found:
        for k in range(len(centers) - 1):
            r0, c0 = centers[k]
            r1, c1 = centers[k + 1]
            cv.line(image_copy, (c0, r0), (c1, r1), color, 5)
    return image_copy


def save_image(path, images):
    n = len(images)
    if n == 1:
        fig, (ax0) = plt.subplots(ncols=1)
        ax0.imshow(images[0].squeeze(), cmap="gray", interpolation="bicubic")
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = plt.subplots(ncols=n, figsize=(4 * n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image.squeeze(), cmap="gray", interpolation="bicubic")
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_training(history, loo_id, lp, block=True):
    def plot():
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 105)
        ax.set_ylim(0, 2)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0) / (y1 - y0))
        ax.plot(history.history["loss"])
        ax.plot(history.history["val_loss"])
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(["Training", "Validation"], loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(lp, "loss%02d.pdf" % (loo_id)))
        plt.show()

    if block:
        plot()
    else:
        multiprocessing.Process(target=plot).start()


def show_image(images, block=True):
    def plot():
        n = len(images)
        if n == 1:
            fig, (ax0) = plt.subplots(ncols=1)
            ax0.imshow(images[0].squeeze(), cmap="gray", interpolation="bicubic")
            ax0.axes.get_xaxis().set_ticks([])
            ax0.axes.get_yaxis().set_visible(False)
        else:
            fig, axes = plt.subplots(ncols=n, figsize=(4 * n, 4))
            for ax, image in zip(axes, images):
                ax.imshow(image.squeeze(), cmap="gray", interpolation="bicubic")
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.show()

    if block:
        plot()
    else:
        multiprocessing.Process(target=plot).start()
