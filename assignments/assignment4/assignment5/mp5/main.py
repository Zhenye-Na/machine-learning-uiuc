import numpy as np
from sklearn import metrics

from model.sklearn_multiclass import sklearn_multiclass_prediction
from model.self_multiclass import MulticlassSVM

if __name__ == '__main__':
    print('Loading data...')
    mnist = np.loadtxt('data/mnist_test.csv', delimiter=',')

    X_train = mnist[:len(mnist)//2, 1:]
    y_train = mnist[:len(mnist)//2, 0].astype(np.int)

    X_test = mnist[len(mnist)//2:, 1:]
    y_test = mnist[len(mnist)//2:, 0].astype(np.int)

    print('Training Sklearn OVR...')
    y_pred_train, y_pred_test = sklearn_multiclass_prediction(
        'ovr', X_train, y_train, X_test)
    print('Sklearn OVR Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))
    print('Sklearn OVR Accuracy (test) :',
          metrics.accuracy_score(y_test, y_pred_test))

    print('Training Sklearn OVO...')
    y_pred_train, y_pred_test = sklearn_multiclass_prediction(
        'ovo', X_train, y_train, X_test)
    print('Sklearn OVO Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))
    print('Sklearn OVO Accuracy (test) :',
          metrics.accuracy_score(y_test, y_pred_test))

    print('Training Sklearn Crammer-Singer...')
    y_pred_train, y_pred_test = sklearn_multiclass_prediction(
        'crammer', X_train, y_train, X_test)
    print('Sklearn Crammer-Singer Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))
    print('Sklearn Crammer-Singer Accuracy (test) :',
          metrics.accuracy_score(y_test, y_pred_test))

    print('Training self OVR...')
    self_ovr = MulticlassSVM('ovr')
    self_ovr.fit(X_train, y_train)
    print('Self OVR Accuracy (train):',
          metrics.accuracy_score(y_train, self_ovr.predict(X_train)))
    print('Self OVR Accuracy (test) :',
          metrics.accuracy_score(y_test, self_ovr.predict(X_test)))

    print('Training self OVO...')
    self_ovo = MulticlassSVM('ovo')
    self_ovo.fit(X_train, y_train)
    print('Self OVO Accuracy (train):',
          metrics.accuracy_score(y_train, self_ovo.predict(X_train)))
    print('Self OVO Accuracy (test) :',
          metrics.accuracy_score(y_test, self_ovo.predict(X_test)))

    print('Training self Crammer-Singer...')
    self_cs = MulticlassSVM('crammer-singer')
    self_cs.fit(X_train, y_train)
    print('Self Crammer-Singer Accuracy (train):',
          metrics.accuracy_score(y_train, self_cs.predict(X_train)))
    print('Self Crammer-Singer Accuracy (test) :',
          metrics.accuracy_score(y_test, self_cs.predict(X_test)))
