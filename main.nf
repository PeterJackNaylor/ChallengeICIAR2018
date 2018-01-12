#!/usr/bin/env nextflow

ExtractResPY = file("ExtractFromResNet.py")

process ExtractFromResNet {
    input:
    file py from ExtractResPY
    output:
    file 'ResNet_Feature.csv' into res_net
    script:
    """
    python $py
    """
}

N_SPLIT = 5
TREE_SIZE = [10, 100, 200, 500, 1000, 10000]
NUMBER_P = ["auto", "log2"]

process TrainRF {
    input:
    file table from res_net
    val n_splits from N_SPLIT
    each n from TREE_SIZE
    each method from NUMBER_P
    output:

    script:
    """
    #!/usr/bin/env python
    from sklearn.model_selection import StratifiedKFold
    from pandas import read_csv, DataFrame
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import numpy as np


    table = read_csv('${table}', index_col=0)
    y = table['y']
    X = table.drop('y', axis=1)
    skf = StratifiedKFold(n_splits=${n_splits})
    val_scores = np.zeros(${n_splits})
    cross = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]
        clf = RandomForestClassifier(n_estimators=${n}, max_features='${method}')
        clf.fit(X_train, y_train)
        print 'Trained model for fold: {}'.format(cross)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        print 'Train Accuracy :: ', accuracy_score(y_train, y_pred_train)
        score_test = accuracy_score(y_test, y_pred_test)
        print 'Test Accuracy  :: ', score_test
        print ' Confusion matrix ', confusion_matrix(y_test, y_pred_test)
        val_scores[cross] = score_test
        cross += 1
    DataFrame(val_scores).to_csv('score.csv')
    """
}
