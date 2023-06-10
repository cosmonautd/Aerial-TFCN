# Traversability Fully Convolutional Network

This repository contains the code, dataset and comparison experiments used in our research
about traversability estimation using a Fully Convolutional Network (FCN).

To run the code, you need to install the dependencies listed in the `requirements.txt` file.
We recommend using a virtual environment to install the dependencies.

```bash
pip install -r requirements.txt
```

The repository is organized as follows:

- `dataset`: contains the dataset used in our experiments. The dataset is organized in
  folders: `X` contains the input images, `Y` contains the ground truth images, `kp` contains
  the keypoints that connect possible traversable regions, `kn` contains the keypoints that
  connect non-traversable regions.
- `comparisons`: contains the code used to compare our method with other methods.
- `weights`: contains the weights of the trained models for the TFCN-RGB and TFCN-G.

## Main code

The `experiments.py` file contains the code used to generate all the results. The `train.py`
file contains the code used to train the models. The `tfcnmodel.py` file contains the code
used to define the TFCN model. The `graphmapx.py` file contains the code used to define the
build the traversability graph. The `pathplanner.py` file contains the code used to generate
paths from the images and keypoints in the dataset.

## Examples

Run the following command to train the TFCN-RGB model without outliers and use image 1 as
test image:

```bash
python train.py --rgb --test 1
```

Run the following command to train the TFCN-G model without outliers and use image 1 as
test image:

```bash
python train.py --grayscale --test 1
```

Run the following command to train the TFCN-RGB model with outliers and use image 1 as
test image:

```bash
python train.py --rgb --test 1 --outliers
```