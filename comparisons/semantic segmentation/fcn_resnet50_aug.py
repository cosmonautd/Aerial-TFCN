# Imports

import torch
import torchvision

import os
import cv2
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"

ap = argparse.ArgumentParser()
ap.add_argument("--test", type=int, default=1, required=False)
ap.add_argument("--outliers", type=bool, default=False, required=False)
args = ap.parse_args()

# Definitions

rgb_classes = [
    [0, 0, 0],  # 0 Unknown
    [255, 0, 0],  # 1 Road
    [0, 0, 255],  # 2 Building/Containers
    [0, 204, 0],  # 3 Grass
    [0, 77, 0],  # 4 Trees/Bushes/Vegetation
    [0, 0, 77],  # 5 Vehicle
]
if args.outliers:
    rgb_classes.append([254, 77, 0])  # 6 Sand
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


def image_generator(trainset_i, loop):
    x_train_i_g = []
    y_train_i_g = []

    for i, data in enumerate(trainset):

        x_train_i, y_train_i = data

        x_train_i = np.rollaxis(x_train_i.cpu().detach().numpy(), 0, 3)
        x_train_i = np.expand_dims(x_train_i, axis=0)

        y_train_i = np.rollaxis(y_train_i.cpu().detach().numpy(), 0, 3)
        y_train_i = np.expand_dims(y_train_i, axis=0)

        for x in range(loop):

            # train_datagen = ImageDataGenerator(
            #         featurewise_center=False,  # set input mean to 0 over the dataset
            #         samplewise_center=False,  # set each sample mean to 0
            #         featurewise_std_normalization=False,  # divide inputs by std of the dataset
            #         samplewise_std_normalization=False,  # divide each input by its std
            #         zca_whitening=False,  # apply ZCA whitening
            #         zoom_range=0.2, # zoom escala
            #         brightness_range=[0.7,1], #brilho
            #         shear_range=0.2, #acho que é angulor
            #         rotation_range=360,  # randomly rotate images in the range (degrees, 0 to 180)
            #         width_shift_range=[-50,50],  # randomly shift images horizontally (fraction of total width) (deslocamento laterial)
            #         height_shift_range=[-50,50],  # randomly shift images vertically (fraction of total height) (deslocamento superio e inferior)
            #         horizontal_flip=True,  # randomly flip images (espelha a imagem)
            #         vertical_flip=True,
            #         fill_mode="nearest"
            # )

            train_datagen = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode="reflect",
                rotation_range=360,
                zoom_range=0.2,
                width_shift_range=0.3,
                height_shift_range=0.3,
                shear_range=30,
            )

            seedInt = random.randint(1, 1000)
            training_set_x = train_datagen.flow(x_train_i, y=None, seed=seedInt)
            training_set_y = train_datagen.flow(y_train_i, y=None, seed=seedInt)

            x_train_i_temp = training_set_x.next()
            y_train_i_temp = training_set_y.next()

            y_train_i_temp = y_train_i_temp.astype(np.byte)

            x_train_i_g.extend(x_train_i_temp)
            y_train_i_g.extend(y_train_i_temp)

    x_train_i_g = np.array(x_train_i_g)
    x_train_i_g = np.rollaxis(x_train_i_g, 3, 1)
    x_train_i_g = torch.from_numpy(x_train_i_g).float()

    y_train_i_g = np.array(y_train_i_g)
    y_train_i_g = np.rollaxis(y_train_i_g, 3, 1)
    y_train_i_g = torch.from_numpy(y_train_i_g).int()

    return (x_train_i_g, y_train_i_g)


# Load dataset

inputs_path = "inputs"
outputs_path = "outputs"

if not os.path.exists(os.path.join(inputs_path, "inputs.pth")):

    inputs = []
    for img in sorted(os.listdir(inputs_path)):
        image = cv2.cvtColor(
            cv2.imread(os.path.join(inputs_path, img)), cv2.COLOR_BGR2RGB
        )
        image = cv2.resize(image, (800, 800))
        inputs.append(image)

    inputs = np.array(inputs)
    inputs = np.rollaxis(inputs, 3, 1)
    inputs = torch.from_numpy(inputs)
    torch.save(inputs, os.path.join(inputs_path, "inputs.pth"))

else:

    inputs = torch.load(os.path.join(inputs_path, "inputs.pth"))

if not os.path.exists(os.path.join(outputs_path, "outputs.pth")):

    outputs = []
    for lab in sorted(os.listdir(outputs_path)):
        label = cv2.cvtColor(
            cv2.imread(os.path.join(outputs_path, lab)), cv2.COLOR_BGR2RGB
        )
        label = cv2.medianBlur(label, 3)
        label = cv2.resize(label, (800, 800))
        label = rgb_to_classes(label)
        outputs.append(label)

    outputs = np.array(outputs)
    outputs = np.rollaxis(outputs, 3, 1)
    outputs = torch.from_numpy(outputs)
    torch.save(outputs, os.path.join(outputs_path, "outputs.pth"))

else:

    outputs = torch.load(os.path.join(outputs_path, "outputs.pth"))

trainset = []
testset = []
test_index = args.test - 1
for n, (i, o) in enumerate(zip(inputs, outputs)):
    if n + 1 == 8 and not args.outliers:
        continue
    if n == test_index:
        testset.append((i, o))
    else:
        trainset.append((i, o))

# Augment data
inputs_gen, outputs_gen = image_generator(trainset, 4)
trainset = torch.utils.data.TensorDataset(inputs_gen, outputs_gen)

# Prepare model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", str(device).upper())

if not os.path.exists("weights"):
    os.makedirs("weights")
w_path = "weights/atpd_weights_%d.pth" % (args.test)

if not os.path.exists(w_path):

    net = torchvision.models.segmentation.fcn_resnet50(num_classes=len(rgb_classes))
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters())

    # https://arxiv.org/abs/1711.05101 AdamW

    print("Training")

    for epoch in range(100):

        print("Started epoch %d" % (epoch + 1))

        train_loss_list = []
        for i, data in enumerate(trainset):
            # get inputs
            inputs, labels = data
            inputs = inputs.to(device).unsqueeze(0).float()
            labels = labels.to(device).long()
            # reset gradients
            optimizer.zero_grad()
            # forward, backward, optimize
            outputs = net(inputs)["out"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        train_loss = np.mean(train_loss_list)
        print(
            "[Map: %2d, Epoch: %3d] Training Loss: %.3f"
            % (args.test, epoch + 1, train_loss)
        )

    print("Finished training")

    torch.save(net.state_dict(), w_path)

print("Testing on map %d" % (args.test))

net = torchvision.models.segmentation.fcn_resnet50(num_classes=len(rgb_classes))
net.to(device)
net.load_state_dict(torch.load(w_path))

image, label = testset[0]
input_rgb = np.rollaxis(image.numpy(), 0, 3)
label = label.squeeze().to(device)

image = image.to(device).unsqueeze(0).float()
output = net(image)["out"][0]
output = output.argmax(0)
output_rgb = classes_to_rgb(output)

show_images([input_rgb, output_rgb])
