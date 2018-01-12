from sklearn.model_selection import StratifiedKFold
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

sp = 3

table = read_csv(name, header=True, index_col=0)
y = table['class']
X = table.drop('class', axis=1)
skf = StratifiedKFold(n_splits=sp)
val_scores = np.zeros(sp)
cross = 0
for train_index, test_index in skf.split(X.as_matrix(), y.as_matrix().flatten()):
    X_train, X_test = X.as_matrix()[train_index], X.as_matrix()[test_index]
    y_train, y_test = y.as_matrix().flatten()[train_index], y.as_matrix().flatten()[test_index]
    clf = RandomForestClassifier(n_estimators=1000)
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