import sklearn.utils
from sklearn.ensemble import RandomForestClassifier
from utility_functions import *
import matplotlib.pyplot as plt


def train_final_forest(ccp_alpha=0.0001):
    '''
    Trains the RandomForestClassifier on the whole preprocesses training set with a chosen ccp_alpha
    Returns: trained model
    Creates Data/predicted.csv from the test set.
    '''
    X, y = pre_process_data()
    X_train, y_train = sklearn.utils.shuffle(X, y)
    model = RandomForestClassifier(ccp_alpha=ccp_alpha, n_jobs=-1)
    model.fit(X_train, y_train)
    create_predictions(model) #creates predict.csv based on the Kaggle test set
    return model

def find_ccp():
    '''
    Searches for the best ccp_alpha.
    Plots Training/Validation accuracy based on each ccp_alpha value.
    Outputs: ccp_alpha that gave highest validation set accuracy.
    '''
    train_score = []
    val_score = []

    x = np.array(range(100)) / 100000.

    for cc in x:
        X_train, X_val, y_train, y_val = prepare_split_data()
        model = RandomForestClassifier(ccp_alpha=cc, n_jobs=-1)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        train_acc = model.score(X_train, y_train)
        train_score.append(train_acc)
        val_score.append(val_acc)
        print("Val: "+ str((val_acc, cc)) + "Train:" + str(train_acc))

    plt.plot(x, train_score, label="Training Set")
    plt.plot(x, val_score, label="Validation Set")
    plt.xlabel('ccp_alpha values')
    plt.ylabel('Classification Accuracy')
    plt.rcParams["axes.titlesize"] = 8
    plt.title('Classification Accuracy on Training Set and Validation Set based on different ccp_alpha values.')
    plt.legend()
    plt.show()

    max_perf = np.argmax(np.array(val_score))
    return x[max_perf] #returns best ccp_alpha







