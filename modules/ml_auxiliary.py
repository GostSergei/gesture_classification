
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from IPython.display import clear_output



def make_X_y(tensor, labels_list):
    X = tensor.reshape(tensor.shape[0], -1)
    y = np.array(labels_list)
    return X, y

def plot_confusion_matrix(y_test, label, label_dict, ax=None):
    # print(classification_report(label, y_test,
    #                             target_names=[l for l in label_dict.values()]))

    conf_mat = confusion_matrix(label, y_test)

    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0, 0, 0.8,  0.8])
    else:
        fig = ax.get_figure()

    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = ax.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                ax.text(j-.2, i+.1, c, fontsize=8)
                
    # cb = fig.colorbar(res)
    cb = plt.colorbar(res, ax=ax)
    ax.set_title('Confusion Matrix')
    _ = ax.set_xticks(range(22), [l for l in label_dict.values()], rotation=90)
    _ = ax.set_yticks(range(22), [l for l in label_dict.values()])
    
def print_classification_report(y_test, label, label_dict):
    print(classification_report(label, y_test,
                                target_names=[l for l in label_dict.values()]))
    
    
def compare_train_test_confusion_matrices(y_train, label_train, y_test, label_test, label_dict, fig_size=[14, 6]):
    fig, (ax_train, ax_test) = plt.subplots(1, 2)
    fig.set_size_inches(fig_size)
    
    plot_confusion_matrix(y_train, label_train, label_dict, ax=ax_train)
    acc_train = accuracy_score(y_train, label_train)
    ax_train.set_title(f"Train: {acc_train :0.4}")
    
    plot_confusion_matrix(y_test, label_test, label_dict, ax=ax_test)
    acc_test = accuracy_score(y_test, label_test)
    ax_test.set_title(f"Test: {acc_test :0.4}")
    
    
def check_KNN(X_train, y_train, X_test, y_test, model_name,  n_start=1, n_end=20):
    acc_train_list = []
    acc_test_list = []
    for n_neighbors in range(n_start, n_end):
        print(f'n_neighbors = {n_neighbors} from {n_end}')
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        label_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, label_train)
        acc_train_list += [acc_train]
        label_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, label_test)
        acc_test_list += [acc_test]
        clear_output()
        
        
    plt.plot(range(n_start, n_end), acc_train_list, label='train')
    plt.plot(range(n_start, n_end), acc_test_list, label='test')
    plt.legend()
    plt.title(model_name)
    plt.gca().set_xlabel('N neighbors')
    plt.gca().set_ylabel('Mean accuracy')
    