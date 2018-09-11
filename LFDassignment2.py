import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             accuracy_score, f1_score, classification_report)


# Load corpus from a file.
# Open the file "corpus_file", return a list of texts and another list
# of all labels. If use_sentiment is true then take the second word in
# line as label (positive or negative), otherwise use the first word as
# label (topics).
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels


# a dummy function that just returns its input
def identity(x):
    return x


def fprint(x, title=''):
    print(title)
    print('='*30)
    print(x)
    print('\n')


def classify(Xtrain, Ytrain, Xtest, Ytest, classifier):
    # let's use the TF-IDF vectorizer
    tfidf = True

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', classifier)])

    # Using the training sets of texts Xtrain and labels Ytrain
    # to train our classifier.
    # import pdb; pdb.set_trace()
    classifier.fit(Xtrain, Ytrain)

    # Classify our test set of text Xtest with our trained classifier.
    Yguess = classifier.predict(Xtest)
    Yproba = classifier.predict_proba(Xtest)
    return Yguess, Yproba


def test_classifiers(classifiers):
    pass


def main():
    # Get list X of texts and Y of labels. Take the first 75% of each list
    # as training sets, and the rest as testing sets.
    X, Y = read_corpus('trainset.txt', use_sentiment=False)
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    # Yguess, Yproba = classify(Xtrain, Ytrain, Xtest, Ytest, KNeighborsClassifier())
    ac_scores = []
    f_scores = []
    for i in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=i)
        Yguess, Yproba = classify(Xtrain, Ytrain, Xtest, Ytest, knn)
        ac_scores.append(accuracy_score(Ytest, Yguess))
        f_scores.append(f1_score(Ytest, Yguess, average='macro'))

    import matplotlib.pyplot as plt

    plt.plot(ac_scores, label='Accuracy Score')
    plt.plot(f_scores, label='F-Score')
    plt.xlim(xmin=1)
    plt.xlabel('K-neighbors')
    plt.ylabel('Score')
    plt.legend()
    plt.show()



    # Test if our classification result meets our actual labels,
    # and print out the result.
    fprint(accuracy_score(Ytest, Yguess), title='Accuracy Score')
    fprint(confusion_matrix(Ytest, Yguess), title='Confusion Matrix')
    fprint(classification_report(Ytest, Yguess), title='Precision, Recall, F-Score and Support (Actual Size)')


if __name__ == '__main__':
    main()
