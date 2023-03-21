from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import matplotlib.pyplot as plt


"""
This file holds the LR, RF, SVM, DT and NB models
"""

def tfidf(df):
    """
    TF-IDF Vectoriser

    Parameters:
        df (Dataframe): The dataset to be vectorised and split into train and test
    Returns:
        x_train, x_test, y_train, y_test (Vectors): The vector representation of train and test, split by data and label 
    """

    # Extracts text and label of the dataset
    x_df = df['text']
    y_df = df['label']

    # Vectorises the data
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(x_df)

    # Fits the data to be split into train and test
    freq_term_matrix = count_vectorizer.transform(x_df)
    tfidf = TfidfTransformer(norm = "l2")
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

    x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix,y_df, random_state=0)

    return x_train, x_test, y_train, y_test, count_vectorizer

def lr(x_train, y_train):
    """
    Logistic Regression

    Parameters:
        x_train (vector): Train data that has been vectorised
        y_train (vectors): Train labels that has been vectorised
    Returns:
        logreg (object): LR model that has been trained on x_train and y_train

    """
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    
    return logreg

def lr_acc(logreg, x_test, y_test):
    """
    Logistic Regression Accuracy

    Parameters:
        logreg (object): LR model that has been trained on x_train and y_train
        x_test (vector): Test data that has been vectorised
        y_test (vectors): Test labels that has been vectorised
    Returns:
        : accuracy of the LR model against x_test and y_test to 3 decimal places
    """

    Accuracy = logreg.score(x_test, y_test)

    return (round(Accuracy,2))

def nb(x_train, y_train):
    """
    Naive Bayes
    
    Parameters:
        x_train (vector): Train data that has been vectorised
        y_train (vectors): Train labels that has been vectorised
    Returns:
        logreg (object): NB model that has been trained on x_train and y_train

    """

    NB = MultinomialNB()
    NB.fit(x_train, y_train)

    return NB

def nb_acc(NB, x_test,  y_test):
    """
    Naive Bayes Accuracy

    Parameters:
        logreg (object): NB model that has been trained on x_train and y_train
        x_test (vector): Test data that has been vectorised
        y_test (vectors): Test labels that has been vectorised
    Returns:
        : accuracy of the NB model against x_test and y_test to 3 decimal places
    """

    Accuracy = NB.score(x_test, y_test)

    return (round(Accuracy*100,2))

def dt(x_train, y_train):
    """
    Decision Tree
    
    Parameters:
        x_train (vector): Train data that has been vectorised
        y_train (vectors): Train labels that has been vectorised
    Returns:
        logreg (object): DT model that has been trained on x_train and y_train

    """

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    return dtc

def dt_acc(dtc, x_test, y_test):
    """
    Decision Tree
    
    Parameters:
        logreg (object): DT model that has been trained on x_train and y_train
        x_test (vector): Test data that has been vectorised
        y_test (vectors): Test labels that has been vectorised
    Returns:
        : accuracy of the DT model against x_test and y_test to 3 decimal places
    """

    Accuracy = dtc.score(x_test, y_test)

    return (round(Accuracy*100,2))

def rf(x_train, y_train):
    """
    Random Forest
    
    Parameters:
        x_train (vector): Train data that has been vectorised
        y_train (vectors): Train labels that has been vectorised
    Returns:
        logreg (object): RF model that has been trained on x_train and y_train

    """

    RF = RandomForestClassifier()
    RF.fit(x_train, y_train)

    return RF

def rf_acc(RF, x_test, y_test):
    """
    Random Forest
    
    Parameters:
        logreg (object): RF model that has been trained on x_train and y_train
        x_test (vector): Test data that has been vectorised
        y_test (vectors): Test labels that has been vectorised
    Returns:
        : accuracy of the RF model against x_test and y_test to 3 decimal places
    """

    Accuracy = RF.score(x_test, y_test)

    return (round(Accuracy*100,2))

def svm(x_train, y_train):
    """
    SVM
    
    Parameters:
        x_train (vector): Train data that has been vectorised
        y_train (vectors): Train labels that has been vectorised
    Returns:
        logreg (object): SVM model that has been trained on x_train and y_train

    """

    SVM = LinearSVC()
    SVM.fit(x_train, y_train)

    return SVM

def svm_acc(SVM, x_test, y_test):
    """
    SVM
    
    Parameters:
        logreg (object): SVM model that has been trained on x_train and y_train
        x_test (vector): Test data that has been vectorised
        y_test (vectors): Test labels that has been vectorised
    Returns:
        : accuracy of the SVM model against x_test and y_test to 3 decimal places
    """

    Accuracy = SVM.score(x_test, y_test)

    return (round(Accuracy*100,2))

def baseline_acc():
    """
    Calculates the baseline accuracy for all 5 models

    Returns:
        : test accuracies for each model in the order: LR, NB, DT, RF and SVM
    """

    df = pd.read_csv('fake_news.csv')
    x_train, x_test, y_train, y_test, cv = tfidf(df)

    return lr_acc(lr(x_train, y_train), x_test, y_test), nb_acc(nb(x_train, y_train), x_test, y_test), dt_acc(dt(x_train, y_train), x_test, y_test), rf_acc(rf(x_train, y_train), x_test, y_test), svm_acc(svm(x_train, y_train), x_test, y_test)
