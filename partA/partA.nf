#!/usr/bin/env nextflow
IMAGE_FOLD = file('../../partA/input')
process MeanCalculation {
    clusterOptions "-S /bin/bash"
    input:
    file fold from IMAGE_FOLD
    output:
    file 'mean_file.npy' into MEAN
    script:
    """
    #!/usr/bin/env python
    import numpy as np
    from glob import glob
    from skimage.io import imread

    photos = glob('$fold/*/*.tif')
    n = len(photos)
    res = np.zeros(shape=3, dtype='float')
    for i, img_path in enumerate(photos):
        img = imread(img_path)
        res += np.mean(img, axis=(0, 1))
    res = res / n
    np.save('mean_file.npy', res)
    """
}

ExtractResPY = file("ExtractFromResNet.py")

IMAGES = file(IMAGE_FOLD + '/*/*.tif')

process ExtractFromResNet {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    input:
    file py from ExtractResPY
    file mean_file from MEAN
    file fold from IMAGE_FOLD
    file img from IMAGES
    output:
    file '*.csv' into res_net
    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py $img $mean_file
    """
}


process Regroup {
    publishDir '../../partA/table', overwrite:true
    clusterOptions "-S /bin/bash -q all.q@compute-0-24"
    input:
    file tbls from res_net .toList()
    output:
    file 'ResNet_Feature.out' into RES
    script:
    """
    grep -v label *.csv | sed -E 's/\\.csv:0//' > ResNet_Feature.out
    """
}

N_SPLIT = 5
TREE_SIZE = Channel.from([10, 100, 200, 500, 1000, 10000])
NUMBER_P = Channel.from(["auto", "log2"])
COMP = Channel.from(15..24)
TREE_SIZE .combine(NUMBER_P) .set{ Param }

process TrainRF {
    publishDir '../../partA/Results', overwrite: true
    clusterOptions "-S /bin/bash -q all.q@compute-0-${key}"
    input:
    file table from RES
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

process RegroupTables {
    publishDir '../../partA/Results', overwrite: true
    clusterOptions "-S /bin/bash"
    input:
    file _ from RF_SCORES .toList()
    output:
    file "all_table.csv" into FINAL_TABLE
    script:
    """
    #!/usr/bin/env python
    from pandas import read_csv, concat
    import numpy as np
    from glob import glob

    files = glob('*.csv')
    l_tab = []
    for f in files:
        table = read_csv(f, index_col=0)
        name, ext = f.split('.')
        __, n_name, p_name = name.split('__')
        table.columns = ['n_{}_{}'.format(n_name, p_name)]
        l_tab.append(table)
    all_tab = concat(l_tab, axis=1)
    all_tab.ix['mean'] = all_tab.mean()
    all_tab.to_csv('all_table.csv')
    """
}

