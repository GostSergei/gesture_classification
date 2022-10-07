
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from IPython.display import clear_output
from tqdm import tqdm



def make_X_y(tensor, labels_list):
    X = tensor.reshape(tensor.shape[0], -1)
    y = np.array(labels_list)
    return X, y

def plot_confusion_matrix(y_test, label, label_dict, ax=None, size=(7, 7)):
    # print(classification_report(label, y_test,
    #                             target_names=[l for l in label_dict.values()]))

    conf_mat = confusion_matrix(y_test, label)
    
    labels_number = len(label_dict.keys())
    
    k_size = labels_number // 22
    
    # print(k_size)

    if ax is None:
        fig = plt.figure(figsize=(size[0]*k_size, size[1]*k_size))
        # ax = fig.add_axes([0, 0, size[0]/6*0.4,  size[1]/6*0.4])
        ax = plt
    else:
        fig = ax.get_figure()

    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]
    
    
    conf_mat = np.array(conf_mat)
    res = ax.imshow(conf_mat, cmap=plt.cm.summer, interpolation='nearest')
    if ax is plt:
        ax = plt.gca()
        
    total_true = np.diag(conf_mat)/conf_mat.sum(axis=1)
    
     
        
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                ax.text(j-.2, i+.1, c, fontsize=8)
                ax.text(width-.2, i+.1, f"{100*total_true[i]:.2f}%", fontsize=8)
                       
    # cb = fig.colorbar(res)
    cb = plt.colorbar(res, ax=ax)
    ax.set_title('Confusion Matrix')
    _ = ax.set_xticks(range(labels_number), [l for l in label_dict.values()], rotation=90)
    _ = ax.set_yticks(range(labels_number), [l for l in label_dict.values()])
    
    return conf_mat
    
def print_classification_report(y_test, label, label_dict):
    print(classification_report(label, y_test,
                                target_names=[l for l in label_dict.values()]))
    
    
def compare_train_test_confusion_matrices(y_train, label_train, y_test, label_test, label_dict, fig_size=[14*2, 6*2]):
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
    
    
    
def test_accuracy(model_pair, tensor, tensor_test, labels_list, labels_list_test, do_scaler=True, n_trials=5, verbose=1): 

    acc_list = []
    print(f"For model: {model_pair[1]}")    
    for i in tqdm(range(n_trials)):
        skip_refitting = i == 0
        model, model_name, X_train, X_test, y_train, y_test = train_model(model_pair, tensor, tensor_test, labels_list,
                                                                          labels_list_test, do_scaler, skip_refitting=skip_refitting)
        label_test = model.predict(X_test)
        acc_test = accuracy_score(y_test, label_test)
        acc_list += [acc_test]
        
    acc_arr = np.array(acc_list)
    if verbose > 0:
        mean_ = acc_arr.mean()*100
        std_ = acc_arr.std()*100
        print(f"Acc of {model_name}: {mean_:.2f}({std_:.2f}) %")
    
    return  acc_arr
    


def check_model(model, model_name, X_train, X_test, y_train, y_test, exp_cofig, label_dict):
    label_train = model.predict(X_train)
    label_test = model.predict(X_test)
    compare_train_test_confusion_matrices(y_train, label_train, y_test, label_test, label_dict)
    plt.gcf().suptitle( model_name+ ": "+ exp_cofig)
    print()
    
    
def _model_is_fitted_(model):
    return not len(dir(model)) == len(dir(type(model)()))

def train_model(model_pair, tensor, tensor_test, labels_list, labels_list_test, do_scaler=True, skip_refitting=False):
    model, model_name = model_pair
    X_train, y_train = make_X_y(tensor, labels_list)
    X_test, y_test = make_X_y(tensor_test, labels_list_test)
    
    
    if model_name in ['STM', 'STMM']:
        new_dims = tensor.shape
        new_dims_test = tensor_test.shape
        if len(new_dims) < 3:
            new_dims = list(new_dims) + [1]
            new_dims_test = list(new_dims_test) + [1]
        X_train = X_train.reshape(new_dims)
        X_test = X_test.reshape(new_dims_test)
        print('Tensor ML model is used!')
    else:
        if do_scaler:
            scaler = StandardScaler().fit(X_train) 
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

    if skip_refitting and _model_is_fitted_(model):
        model = model
        # print('Model fitting was skipped!')
    else:
        model = model.fit(X_train, y_train)
        
    return model, model_name, X_train, X_test, y_train, y_test



    