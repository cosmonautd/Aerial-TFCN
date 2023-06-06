import torch
import torchvision

import os
import cv2
import numpy as np

classes = [
    'Unknown',
    'Road',
    'Building/Containers',
    'Grass',
    'Trees/Bushes/Vegetation',
    'Vehicle'
]

rgb_classes = [
    [  0,   0,   0], # 0 Unknown
    [255,   0,   0], # 1 Road
    [  0,   0, 255], # 2 Building/Containers
    [  0, 204,   0], # 3 Grass
    [  0,  77,   0], # 4 Trees/Bushes/Vegetation
    [  0,   0,  77], # 5 Vehicle
]

def rgb_to_classes(rgb_label):
    h, w, d = rgb_label.shape
    class_label = np.zeros((h, w, 1))
    for y in range(h):
        for x in range(w):
            pixel = rgb_label[y][x]
            class_ = np.argmin(np.linalg.norm(pixel - rgb_classes, axis=1))
            class_label[y][x] = class_
    return class_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = {
    0: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    1: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    2: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    3: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    4: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
    5: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
}

accuracy = np.zeros(6)
precision = np.zeros(6)
recall = np.zeros(6)
f1_score = np.zeros(6)

for k in range (7):

    w_path = 'weights/atpd_weights_%d.pth' % (k+1)
    net = torchvision.models.segmentation.fcn_resnet50(num_classes=len(rgb_classes))
    net.to(device)
    net.load_state_dict(torch.load(w_path))
    
    image = cv2.imread(os.path.join('inputs', 'aerial%02d.jpg' % (k+1)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rollaxis(image, 2, 0)
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)

    label = cv2.imread(os.path.join('outputs', 'aerial%02d.jpg' % (k+1)))
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    label = cv2.medianBlur(label, 3)
    label = rgb_to_classes(label)
    label = torch.from_numpy(label).long()
    label.squeeze_()
    
    image = image.to(device)
    output = net(image)['out'][0]
    output = output.argmax(0)
    output = output.cpu()

    for i in range(label.size(0)):
        for j in range(label.size(1)):
            true_class = label[i,j].item()
            predicted_class = output[i,j].item()
            if predicted_class == true_class:
                data[true_class]['tp'] += 1
                for c in [c for c in data.keys() if c != true_class]:
                    data[c]['tn'] += 1
            elif predicted_class != true_class:
                data[true_class]['fn'] += 1
                data[predicted_class]['fp'] += 1
                for c in [c for c in data.keys() if c != true_class and c != predicted_class]:
                    data[c]['tn'] += 1

for c in range(len(rgb_classes)):
    d = data[c]
    accuracy[c] = (d['tp'] + d['tn'])/(d['tp'] + d['tn'] + d['fp'] + d['fn'])
    precision[c] = d['tp']/(d['tp'] + d['fp'])
    recall[c] = d['tp']/(d['tp'] + d['fn'])
    f1_score[c] = 2*(precision[c]*recall[c])/(precision[c] + recall[c])

    print('Class: %s' % (classes[c]))
    # print('Accuracy: %.3f' % accuracy[c])
    print('Precision: %.3f' % precision[c])
    print('Recall: %.3f' % recall[c])
    print('F1 Score: %.3f' % f1_score[c])
    print()

# print('Average Accuracy: %.3f' % (np.mean(accuracy)))
print('Average Precision: %.3f' % (np.mean(precision)))
print('Average Recall: %.3f' % (np.mean(recall)))
print('Average F1 Score: %.3f' % (np.mean(f1_score)))