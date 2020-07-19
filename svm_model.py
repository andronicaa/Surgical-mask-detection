from sklearn.svm import SVC
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import librosa, librosa.display
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix



# function for reading
def read_from_file(path):

    names = [] # array for the names of the audio files
    data = [] # array for extracted features
    for p in path:
        for filepath in glob.glob(p):
            # rest_type for a high quality reading
            signal, sample_rate = librosa.load(filepath, res_type = 'kaiser_fast')
            
            names.append(os.path.basename(filepath))
            # extract features from each record
            # this is a matrix and that's why i make that mean on every line
            mfcc_feature = librosa.feature.mfcc(y = signal, sr = sample_rate,
                        n_mfcc=40)
            # plot the mfccs
            # librosa.display.specshow(mfcc_feature, sr = sample_rate, hop_length = 512)
            # plt.xlabel("Time")
            # plt.ylabel("MFCC coefficients")
            # plt.colorbar()
            # plt.title("MFCCs feature")
            # plt.show()
            get_feature = np.mean(mfcc_feature.T, axis = 0)
            data.append(get_feature)

    # convert in np.array
    data = np.array(data)
    # return a tuple consisting of the file names and the extracted data
    return (names, data)

# function that get labels for every audio file from train and validation dataset
def get_labels(path, names):

    # count the lines in the file
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


    model = SVC()
    model.fit(train_data, train_labels)
    validation_predictions = model.predict(validation_data)
    # getting prediction on the test dataset
    test_predictions = model.predict(test_data)

    # print precision and recall
    print(classification_report(validation_labels, validation_predictions, target_names=["0", "1"]))
    titles = [("Confusion matrix, without normalization", None),
                ("Normalized confusion matrix", 'true')]
    for title, normalize in titles:
        display = plot_confusion_matrix(model, validation_data, validation_labels,
            cmap = plot.cm.Blues, normalize = normalize)
        display.ax_.set_title(title)
        print(title)
        print(display.confusion_matrix)
    plt.show()
    g = open("submission.txt", "w")
    g.write("name,label\n")
    for i in range(len(test_names)):
        g.write(f'{test_names[i]},{test_predictions[i]}\n')
    g.close()
    print(np.mean(validation_predictions == validation_labels))


if __name__ == "__main__":
    get_dataset()