

IMAGE_FOLD = file("../../partA/input/*/*.tif")
WEIGHT_NUC = file("../../segmentation_table/metadata/DIST__16_0.00005_0.001")
MEAN_FILE = file("../../segmentation_table/metadata/mean_file.npy")
SEGMENT = file("PredictFromDist.py")

process segment {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    publishDir "../../segmentation_table/segmentation", overwrite: true
    input:
    file py from SEGMENT
    file mean_file from MEAN_FILE
    file weight from WEIGHT_NUC
    file img from IMAGE_FOLD
    output:
    file "out/C_*.png" 
    file "out/dist_*.png"
    set file(img), file("out/pred_*.png") into RGB_AND_PRED
    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py $img $weight $mean_file
    """
}

TABLE = file("CreateTable.py")

process create_table {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    publishDir "../../segmentation_table/table", overwrite: true
    input:
    file py from TABLE
    set file(rgb), file(pred) from RGB_AND_PRED
    output:
    file "*.csv" TABLE_PER_IMAGE
    script:
    """
    python $py $rgb $pred
    """
}

SUMTABLE = file("SumTable.py")

process sum_tabs {
    clusterOptions "-S /bin/bash"
    queue "all.q"
//    publishDir "../../segmentation_table/", overwrite: true
    input:
    file py from TABLE
    file csv_tab TABLE_PER_IMAGE
    output:
    file "new_${csv_table}" into DES_TAB
    """
    python $py $csv_tab new_${csv_table}
    """
}

process regroup_desc_tab {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    publishDir "../../segmentation_table/Descriptive", overwrite: true
    input:
    file _ from DES_TAB .toList()
    output:
    file "nuclei_description.out" into TAB_DESC
    """
    grep -v label *.csv | sed -E 's/\\.csv:0//' > nuclei_description.out
    """
}

N_SPLIT = 5
TREE_SIZE = Channel.from([10, 100, 200, 500, 1000, 10000])
NUMBER_P = Channel.from(["auto", "log2"])
COMP = Channel.from(15..24)
TREE_SIZE .combine(NUMBER_P) .set{ Param; Param2 }

process only_nuc_RF {
    publishDir '../../segmentation_table/Results', overwrite: true
    clusterOptions "-S /bin/bash -q all.q@compute-0-${key}"
    input:
    file table from TAB_DESC
    val n_splits from N_SPLIT
    set n, method from Param
    val key from COMP
    output:
    file "score__${n}__${method}.csv" into RF_SCORES
    script:
    """
    #!/usr/bin/env python
    from sklearn.model_selection import StratifiedKFold
    from pandas import read_csv, DataFrame
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import numpy as np


    table = read_csv('${table}', header=None, index_col=0)
    y = table[1]
    X = table.drop(1, axis=1)
    skf = StratifiedKFold(n_splits=${n_splits}, shuffle=True, random_state=42)
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
    DataFrame(val_scores).to_csv('score__${n}__${method}.csv')
    """

}

RESNET_FEAT = file("../../partA/table/ResNet_Feature.out")

process all_RF {
    publishDir '../../segmentation_table/Results', overwrite: true
    clusterOptions "-S /bin/bash -q all.q@compute-0-${key}"
    input:
    file table from TAB_DESC
    val n_splits from N_SPLIT
    set n, method from Param
    val key from COMP
    output:
    file "score__${n}__${method}.csv" into RF_SCORES
    script:
    """
    #!/usr/bin/env python
    from sklearn.model_selection import StratifiedKFold
    from pandas import read_csv, DataFrame
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import numpy as np


    table = read_csv('${table}', header=None, index_col=0)
    y = table[1]
    X = table.drop(1, axis=1)
    skf = StratifiedKFold(n_splits=${n_splits}, shuffle=True, random_state=42)
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
    DataFrame(val_scores).to_csv('score__${n}__${method}.csv')

}