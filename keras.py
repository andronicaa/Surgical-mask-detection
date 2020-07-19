import glob
import numpy as np
import os
import librosa, librosa.display
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Dense
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import GaussianDropout
from keras.regularizers import l2


# functie pentru citire
def read_from_file(path):
    names = []
    data = []
    for p in path:
        for filepath in glob.glob(p):
            fs, data_file = librosa.load(filepath, res_type = 'kaiser_fast')
            names.append(os.path.basename(filepath))
            get_feature = np.mean(librosa.feature.mfcc(y = fs, sr=data_file,
                        n_mfcc = 128).T, axis = 0)
            data.append(get_feature)

    data = np.array(data)
   
    return (names, data)

def get_labels(path, names):
   
    nr_linii = 0
    for file in path:
        for line in open(file): nr_linii += 1
    
    
    labels = [0] * nr_linii
    for p in path:
        labels_file = open(p, 'r')
        for line in labels_file.readlines():
            name = line.split(',')[0]
            if name in names:
                labels[names.index(name)] = (
                    int(line.split(',')[1])
                )
    
    
    labels = np.array(labels)
    return labels


def verif_overfitting(history):
    
    figure, axis = plt.subplots(2)
    # graficul pentru acuratete
    axis[0].plot(history.history["accuracy"], label="Acuratete date Train")
    axis[0].plot(history.history["val_accuracy"], label="Acuratete date validare")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    # graficul pentru eroare
    axis[1].plot(history.history["loss"], label="Eroare train")
    axis[1].plot(history.history["val_loss"], label="Eroare validare")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    plt.show()


def get_dataset():


    train_names = []
    train_data = []
    validation_names = []
    validation_data = []
    test_names = []
    test_data = []

   
    train_names, train_data = read_from_file(['./ml-fmi-23-2020/train/train/*'])
    validation_names, validation_data = read_from_file(['./ml-fmi-23-2020/validation/validation/*'])
    train_labels = get_labels(['./ml-fmi-23-2020/train.txt'], train_names)
    validation_labels = get_labels(['./ml-fmi-23-2020/validation.txt'], validation_names)
    test_names, test_data = read_from_file(['./ml-fmi-23-2020/test/test/*'])


    # the sequential allows us to create models layer-by-layer
    # I'll make 4 layers(2 hidden) with activation relu and kernel_regularization l2(after testing this value brought me the best accuracy)
    model = Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(GaussianDropout(0.1))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(GaussianDropout(0.1))
    # on the output layer are 2 neurons because the classification is binary and the activation softmax because that it help us to decide what is the label for the sample(if we sum the probabilities on the last layer the result will be 1 and the higher probability will be chosen)
    model.add(layers.Dense(2, activation='softmax'))
    # sparse_categorical_crossentropy is for mutually exclusive(classes)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    # epoch - one pass over the entire dataset(I chose to train tha data set on epochs = 100)
    # batch_size - number of samples that will be propagated through the network
    history = model.fit(train_data, train_labels, epochs = 100, batch_size = 128)
    # the plot with accuracy the error
    verif_overfitting(history)
    # after training we evaluate the accuracy on the validation data
    validation_accuracy = model.evaluate(validation_data, validation_labels, verbose = 0)
    print(validation_accuracy)
    test_predictions = model.predict_classes(test_data)
    g = open("submission.txt", "w")
    g.write("name,label\n")
    for i in range(len(test_names)):
        g.write(f'{test_names[i]},{test_predictions[i]}\n')
    g.close()
   


if __name__ == "__main__":
    get_dataset()