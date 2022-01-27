import sklearn.utils
from sklearn.ensemble import RandomForestClassifier
from utility_functions import *
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    Source: https://stackoverflow.com/a/57178527
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class LogisticModel:

    def __init__(self, features, target, normalize=False):
        '''
        Initializes the Model.
        In case of multiclass classification, it converts the targets and uses one-vs-all paradigm
        '''
        self.unique_labels = np.unique(target)
        self.one_against_all_targets = []

        if self.unique_labels.size > 2:
            self.number_of_models = self.unique_labels.size
            for label in self.unique_labels:
                label_train = np.copy(target)
                similarity_mask = label_train == label
                difference_mask = label_train != label
                label_train[difference_mask] = 0
                label_train[similarity_mask] = 1
                self.one_against_all_targets.append(label_train)
        else:
            self.number_of_models = 1
            self.one_against_all_targets.append(target)

        self.features = features
        self.target = target
        self.normalize = normalize

        if normalize:
            self.mean = np.mean(self.features, 0)
            self.std = np.std(self.features, 0)
            self.features = self.normalize_data(self.features)

        self.features = np.column_stack((np.ones(self.features.shape[0]), self.features)) #add bias term X_0 of ones

    def normalize_data(self, features):
        return (features - self.mean) / self.std

    def fit(self, rate, lam, num_iter):
        '''
        :rate: learning rate
        :lam: regularization lambda weight
        :num_iter: number of iterations of the learning loop
        '''
        self.hypothesis = []
        for i in range(self.number_of_models):
            target = self.one_against_all_targets[i]
            w_k = np.zeros(self.features.shape[1])
            for j in range(num_iter):
                w_k = w_k + rate / len(target) * (((target - sigmoid(self.features @ w_k)) @ self.features) + lam * np.sign(w_k))
            self.hypothesis.append(w_k)

    def predict(self, features):
        try:
            if self.normalize:
                features = self.normalize_data(features)
            features = np.column_stack((np.ones(features.shape[0]), features)) #adds bias term
            return sigmoid(features @ self.hypothesis[0]) > 0.5
        except AttributeError:
            print("LR.Predict Error: no existing learned hypothesis TIP: run LR.fit before predicting.")
        except ValueError:
            print("LR.Predict Error: features dimension is incorrect.")

    def multiclass_predict(self, features):
        try:
            if self.normalize:
                features = self.normalize_data(features)
            features = np.column_stack((np.ones(features.shape[0]), features)) #adds bias term
            class_probabilities = []
            for w_k in self.hypothesis:
                class_probabilities.append(sigmoid(features @ w_k))
            return self.unique_labels[np.argmax(np.array(class_probabilities).T, axis=1)]
        except AttributeError:
            print("LR.Predict Error: no existing learned hypothesis TIP: run LR.fit before predicting.")
        except ValueError:
            print("LR.Predict Error: features dimension is incorrect.")

    def evaluate_acc(self, X, y):
        if self.number_of_models == 1:
            return np.mean(self.predict(X) == y)
        else:
            return np.mean(self.multiclass_predict(X) == y)

    def k_fold(self, k, rate, num_iter):
        k_features = np.array_split(self.features[:, 1:], k) #everything except bias term, it gets added later
        k_targets = np.array_split(self.target, k)
        err = 0
        for i in range(k):
            ith_features = np.concatenate(np.delete(k_features, i, 0))
            ith_targets = np.concatenate(np.delete(k_targets, i, 0))
            ith_model = LogisticModel(ith_features, ith_targets)
            ith_model.fit(rate, num_iter)
            err += ith_model.evaluate_acc(k_features[i], k_targets[i])
        return (1/k)*err

def train_final_logistic(lr=0.2, reg=0.01, iter=1000):
    '''
    Trains the RandomForestClassifier on the whole preprocesses training set with a chosen ccp_alpha
    Returns: trained model
    Creates Data/predicted.csv from the test set.
    '''
    X, y = pre_process_data()
    X_train, y_train = sklearn.utils.shuffle(X, y)
    model = LogisticModel(X_train, y_train, normalize=True)
    model.fit(lr, reg, iter)

    test = pd.read_csv('Data/test.csv', index_col='S.No')
    test = extract_features(test, True)
    test_np = test.to_numpy()
    predicted = model.multiclass_predict(test_np).astype(int)
    print(np.bincount(predicted))
    predicted_df = pd.DataFrame(predicted, columns=['LABELS'])
    predicted_df.index.name = 'S.No'
    predicted_df.to_csv('Data/predicted.csv')
    return model

def search_learning_rate(reg=0, iter=10000):
    '''
    Searches for the best learning rate.
    Plots Training/Validation accuracy based on each learning rate value.
    Outputs: learning rate that gave highest validation set accuracy.
    '''
    train_score = []
    val_score = []

    x = np.array(range(50)) / 100. + 0.01

    for lr in x:
        X_train, X_val, y_train, y_val = prepare_split_data(smote=False)
        model = LogisticModel(X_train, y_train, normalize=True)
        model.fit(lr, reg, iter)
        val_acc = model.evaluate_acc(X_val, y_val)
        train_acc = model.evaluate_acc(X_train, y_train)
        train_score.append(train_acc)
        val_score.append(val_acc)
        print("Val: " + str((val_acc, lr)) + "Train:" + str(train_acc))

    plt.plot(x, train_score, label="Training Set")
    plt.plot(x, val_score, label="Validation Set")
    plt.xlabel('Learning Rate Values')
    plt.ylabel('Classification Accuracy')
    plt.rcParams["axes.titlesize"] = 8
    plt.title('Classification Accuracy of Logistic Regression on Training Set and Validation Set based on different learning rate values.')
    plt.legend()
    plt.show()

    max_perf = np.argmax(np.array(val_score))
    return x[max_perf]  # returns best ccp_alpha

def search_ridge(lr=0.2, iter=1000):
    '''
    Searches for the best regularization weight.
    Plots Training/Validation accuracy based on each regularization weight value.
    Outputs: regularization weight that gave highest validation set accuracy.
    '''
    train_score = []
    val_score = []

    x = np.array(range(50)) / 100. + 0.01
    x = np.concatenate((x, np.array(range(10))), axis=None)

    for reg in x:
        X_train, X_val, y_train, y_val = prepare_split_data(smote=False)
        model = LogisticModel(X_train, y_train, normalize=True)
        model.fit(lr, reg, iter)
        val_acc = model.evaluate_acc(X_val, y_val)
        train_acc = model.evaluate_acc(X_train, y_train)
        train_score.append(train_acc)
        val_score.append(val_acc)
        print("Val: " + str((val_acc, reg)) + "Train:" + str(train_acc))

    plt.scatter(x, train_score, label="Training Set")
    plt.scatter(x, val_score, label="Validation Set")
    plt.xlabel('Learning Rate Values')
    plt.ylabel('Classification Accuracy')
    plt.rcParams["axes.titlesize"] = 8
    plt.title(
        'Classification Accuracy of LR on Training Set and Validation Set based on different regularization weights.')
    plt.legend()
    plt.show()

    max_perf = np.argmax(np.array(val_score))
    return x[max_perf]  # returns best ccp_alpha

def search_iters(lr=0.2, reg=0):
    '''
    Searches for the best number of iterations.
    Plots Training/Validation accuracy based on each number of iterations.
    Outputs: number of iterations that gave highest validation set accuracy.
    '''
    train_score = []
    val_score = []
    i = [10, 100, 1000, 2000, 5000, 10000]
    x = np.array(i)

    for iter in x:
        X_train, X_val, y_train, y_val = prepare_split_data(smote=False)
        model = LogisticModel(X_train, y_train, normalize=True)
        model.fit(lr, reg, iter)
        val_acc = model.evaluate_acc(X_val, y_val)
        train_acc = model.evaluate_acc(X_train, y_train)
        train_score.append(train_acc)
        val_score.append(val_acc)
        print("Val: " + str((val_acc, iter)) + "Train:" + str(train_acc))

    fig, ax = plt.subplots()
    rects1 = ax.bar(np.arange(len(i)) - 0.35/2, train_score, 0.35, label="Training Set")
    rects2 = ax.bar(np.arange(len(i)) + 0.35/2, val_score, 0.35, label="Validation Set")

    ax.set_ylabel('Classification Accurac')
    ax.set_xlabel('Number Of Iterations')
    plt.rcParams["axes.titlesize"] = 9
    ax.set_title(
        'Classification Accuracy of LR on Training Set and Validation Set based on number of iterations.')
    ax.set_xticks(np.arange(len(i)))
    ax.set_xticklabels(i)
    ax.legend(loc='lower right')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    max_perf = np.argmax(np.array(val_score))
    return x[max_perf]  # returns best ccp_alpha
