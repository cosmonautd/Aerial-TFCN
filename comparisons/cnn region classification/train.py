import os
import cv2
import numpy as np
import sklearn.model_selection
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

import sys
sys.path.append('../../')

import model
import toolbox

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import google.colab
    running = 'colab'
    colab = [running]
except:
    running = 'local'
    colab = []

if running in colab:
    from google.colab import drive, auth
    drive.mount('/content/drive')
    root = 'drive/My Drive/Colab Notebooks/TFCN'
else: 
    root = '.'

ap = argparse.ArgumentParser()
ap.add_argument('--test', type=int, default=1, required=False, help='Select a map to use as test')
ap.add_argument('--use', type=int, default=8, required=False, help='Select how many images to use')
args = ap.parse_args()

X_path = os.path.join(root, '../../dataset/X')
Y_path = os.path.join(root, './classes')

# Criação de diretório para armazenar os pesos treinados
W_path = os.path.join(root, 'weights%d' % (args.use))
if not os.path.exists(W_path):
    os.makedirs(W_path)

X_img = list()
C_img = dict()

resolution = 15
if args.use == 8:
    classes = ['road', 'building', 'grass', 'vegetation', 'vehicle', 'sand']
    cost_vector = [-100, 100, -80, 100, 80, -50]
elif args.use == 7:
    classes = ['road', 'building', 'grass', 'vegetation', 'vehicle']
    cost_vector = [-100, 100, -80, 100, 80]

# Obtenção das camadas representativas das classes
for i, imagename in enumerate(sorted(os.listdir(X_path))[:args.use]):
    if imagename.endswith('.jpg'):
        x = toolbox.imread(os.path.join(X_path, imagename), color='rgb')
        X_img.append(x)
        C_img[i] = dict()
        for j, class__ in enumerate(classes):
            c_path = os.path.join(Y_path, imagename[:-4]+'-'+class__+'.jpg')
            if os.path.exists(c_path): c = toolbox.imread(c_path, color='grayscale')
            else: c = np.zeros((x.shape[0], x.shape[1]))
            _, c = cv2.threshold(c, 5, 255, cv2.THRESH_BINARY)
            C_img[i][class__] = c

cross_val = sklearn.model_selection.LeaveOneOut()
cross_val.get_n_splits(X_img)

# Total de amostras no dataset original
N = len(X_img)

# Percorre as divisões de conjuntos de treino e teste
# Leave One Out
for train_index, test_index in cross_val.split(X_img):

    if test_index + 1 != args.test:
        continue

    # Extração das regiões e rótulos para treinamento

    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test = list()

    for i in train_index:
        x = X_img[i]
        grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
        for g in grid:
            [tlx, tly, r] = g
            region = x[tly:tly+r, tlx:tlx+r]
            if region.shape != (resolution,resolution,3): continue
            output = np.zeros(len(classes))
            for j, class__ in enumerate(classes):
                c = C_img[i][class__]
                c_region = c[tly:tly+r, tlx:tlx+r]
                output[j] = (np.sum(c_region)/(resolution**2))/255
            X_train.append(region)
            Y_train.append(output)

    X_train = np.array(X_train).astype(np.float32)
    Y_train = np.array(Y_train).astype(np.float32)

    # Threshold
    Y_train[Y_train > 0.15] = 1
    Y_train[Y_train <= 0.15] = -1

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

    X_val = X_train[:int(0.2*len(X_train))]
    Y_val = Y_train[:int(0.2*len(Y_train))]

    X_train = X_train[int(0.2*len(X_train)):]
    Y_train = Y_train[int(0.2*len(Y_train)):]

    # Treinamento da CNN
    # Função para construção da CNN
    def build_cnn():
        # Organização sequencial de camadas
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=5, padding='same', activation='tanh', input_shape=(resolution,resolution,3)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=50, kernel_size=5, padding='same', activation='tanh'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(len(classes), activation='tanh'))
        
        # Compilação do modelo. Definição da função de perda e algoritmo de treinamento.
        model.compile(loss='mse', optimizer='adam')
        return model

    # Construção da CNN
    CNN = build_cnn()

    weights_path = os.path.join(W_path, 'weights_%02d.hdf5' % (test_index + 1))

    # Definição de um callback para salvamento da melhor combinação de pesos
    checkpointer = ModelCheckpoint(weights_path, verbose=1, save_best_only=True)

    if not os.path.exists(weights_path):
        # Treinamento da CNN
        hist_cnn = CNN.fit(X_train, Y_train, batch_size=32, epochs=100, callbacks=[checkpointer],
                            validation_data=(X_val, Y_val))

    # Carregamento da melhor combinação de pesos obtida durante o treinamento
    CNN.load_weights(weights_path)

    thresholds = list()
    outputs = CNN.predict(X_train)
    for j, class__ in enumerate(classes):
        present = np.where(Y_train[:,j] == 1)[0]
        not_present = np.where(Y_train[:,j] == -1)[0]
        present = outputs[present][:,j]
        not_present = outputs[not_present][:,j]
        print()
        print('Class %s present: %.3f (+/- %.3f)' % (classes[j], np.mean(present), np.std(present)))
        print('Class %s not present: %.3f (+/- %.3f)' % (classes[j], np.mean(not_present), np.std(not_present)))

        import statsmodels.api as sm

        present_pdf = scipy.stats.gaussian_kde(present)
        present_cdf = sm.distributions.ECDF(present)
        not_present_pdf = scipy.stats.gaussian_kde(not_present)
        not_present_cdf = sm.distributions.ECDF(not_present)

        x_eval = np.linspace(-1.7, 1.7, num=1000)

        # plt.plot(x_eval, present_pdf(x_eval), color='g', linestyle='-', label='%s present PDF' % (classes[j].capitalize()))
        # plt.plot(x_eval, present_cdf(x_eval), color='g', linestyle=':', label='%s present CDF' % (classes[j].capitalize()))
        # plt.plot(x_eval, not_present_pdf(x_eval), color='r', linestyle='-', label='%s present PDF' % (classes[j].capitalize()))
        # plt.plot(x_eval, 1-not_present_cdf(x_eval), color='r', linestyle=':', label='%s present CCDF' % (classes[j].capitalize()))

        threshold = None
        for x in x_eval:
            if 1-not_present_cdf(x) <= present_cdf(x):
                threshold = x
                break
        
        thresholds.append(threshold)

        print('Threshold: %.3f' % threshold)
        
        # plt.axvline(threshold, 0, 100, linestyle=':', label='Threshold')

        # plt.legend()
        # plt.show()
    
    np.savetxt(os.path.join(W_path, 'thresholds-%d.txt' % (test_index + 1)), thresholds, fmt='%f')

    # Teste
    y_img = np.zeros((1000,1000))
    x = X_img[test_index[0]]
    grid = toolbox.grid_list_overlap(x, resolution, overlap=0.4)
    for g in grid:
        [tlx, tly, r] = g
        region = x[tly:tly+r, tlx:tlx+r]
        if region.shape != (resolution,resolution,3): continue
        y = CNN.predict(np.array([region]))[0]
        for k, _ in enumerate(y):
            d = y[k] - thresholds[k]
            if d > 0:
                y[k] = 0.5 + min(0.5, 1.5*d)
            elif d < 0:
                y[k] = 0.5 + max(-0.5, 1.5*d)
            else:
                y[k] = 0.5
        cost = y @ cost_vector
        y_img[tly:tly+r, tlx:tlx+r] = (y_img[tly:tly+r, tlx:tlx+r] + cost)/2
    
    y_img = (y_img - np.min(y_img))/np.max(y_img)
    # toolbox.show_image([y_img])
    