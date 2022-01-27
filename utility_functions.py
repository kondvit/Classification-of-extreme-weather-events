import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def normalize_data(features):
    return (features - np.mean(features, 0)) / np.std(features, 0)

def create_predictions(model):
    '''
    Creates predicted.csv for Kaggle submissions.
    '''
    test = pd.read_csv('Data/test.csv', index_col='S.No')
    test = extract_features(test, True)
    test_np = test.to_numpy()
    predicted = model.predict(test_np).astype(int)
    print(np.bincount(predicted))
    predicted_df = pd.DataFrame(predicted, columns=['LABELS'])
    predicted_df.index.name = 'S.No'
    predicted_df.to_csv('Data/predicted.csv')

def make_report(predictions, y_val):
    '''
    Creates a report with a confusion matrix heatmap
    '''
    cm = confusion_matrix(y_val, predictions)
    plt.matshow(cm)
    for (x, y), value in np.ndenumerate(cm):
        plt.text(x, y, f"{value}", va="center", ha="center")
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.show()
    print(cm)
    print(classification_report(y_val, predictions))
    print(accuracy_score(y_val, predictions))

def extract_features(data, isTest=False):
    '''
    Transforms 'time' column with format yyyymmdd
    into
    'year' with format yyyy
    and
    'month/day' with format mmdd
    '''
    time_df = pd.DataFrame()
    years = (data['time']/10000).astype(int)
    months_and_days = (data['time']-10000*years).astype(int)
    days = data['time']%100
    time_df = pd.DataFrame({'year':years, 'month/day':months_and_days})
    if isTest:
        a = pd.concat([data.iloc[:, :-1], time_df], axis=1)
    else:
        a = pd.concat([data.iloc[:,:-2], time_df], axis=1)
        a = pd.concat([a, data.iloc[:,-1]], axis=1)
    return a

def pre_process_data():
    '''
        Preprocesses the data:
            Drops duplicates
            Splits the time column into two columns
            Splits the data into training and validation set
            Oversamples the training set for unbalanced classes
    '''
    df = pd.read_csv('Data/train.csv', index_col='S.No')
    df = df.drop_duplicates()
    df = extract_features(df, isTest=False)
    np_df = df.to_numpy()
    X = np_df[:, :-1]
    y = np_df[:, -1]
    return X, y

def prepare_split_data(smote=True):
    '''
    This function was used for model selection

    Returns: pre-processes the data and returns train/val data split.
    '''
    #Load and pre-process the data.
    X, y = pre_process_data()

    #Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    #Oversample unbalanced classes (Note: we don't oversample validation and test set, they are kept unbalanced)
    if smote:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    return X_train, X_val, y_train, y_val

def prep_data_and_classify_datapoints(k=5):
    '''
    Attempt to classify data points with same features but different labels.
    '''
    df = pd.read_csv('Data/train.csv', index_col='S.No')
    df = extract_features(df, isTest=False)

    #Step 1 drop RAW duplicates
    df1 = df.drop_duplicates()

    #Step 2 drop all feature duplicates keep NONE
    df2 = df1.drop_duplicates(subset=df.columns[0:-1], keep=False)

    #Step 3 Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    df_knn_np = df2.to_numpy()
    knn.fit(normalize_data(df_knn_np[:, :-1]), df_knn_np[:, -1])

    mask = df1.duplicated(subset=df.columns[0:-1], keep='first')
    dups_np = pd.DataFrame(df1[mask]).to_numpy()[:, :-1]
    predictions = knn.predict(normalize_data(dups_np))
    final_np = np.column_stack((dups_np, predictions))

    predicted_df = pd.DataFrame(final_np, columns=df.columns)

    final_df = pd.concat([predicted_df, df2], ignore_index=True)
    return final_df