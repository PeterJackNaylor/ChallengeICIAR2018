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
        img = imread(img_path).astype('uint8')
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
    file '*.npy' into res_net
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
    file 'res_untouched.npy' into RES, RES2
    script:
    """
    #!/usr/bin/env python
    import numpy as np
    from glob import glob
    files = glob('*.npy')
    size = np.load(files[0]).shape[0]
    resnet = np.zeros((len(files), size), dtype='float')
    for k, f in enumerate(files):
        resnet[k] = np.load(f)
    np.save('res_untouched.npy', resnet)
    """
}

ExtractResPYTrained = file("ExtractFromTrainedResNet.py")
Trained_files = file("../../partA/trainedmodel/fromCV/0.0001__0.99__0.00005_fold_0.h5")

process ExtractFromTrainedResNet {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    input:
    file py from ExtractResPYTrained
    file mean_file from MEAN
    file fold from IMAGE_FOLD
    file img from IMAGES
    file weights from Trained_files
    output:
    file '*.npy' into res_trainednet
    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py $img $weights $mean_file
    """
}

process RegroupTrained {
    publishDir '../../partA/table', overwrite:true
    clusterOptions "-S /bin/bash -q all.q@compute-0-24"
    input:
    file tbls from res_trainednet .toList()
    output:
    file 'res_traineduntouched.npy' into RESTRAINED, RESTRAINED2
    script:
    """
    #!/usr/bin/env python
    import numpy as np
    from glob import glob
    files = glob('*.npy')
    size = np.load(files[0]).shape[0]
    resnet = np.zeros((len(files), size), dtype='float')
    for k, f in enumerate(files):
        resnet[k] = np.load(f)
    np.save('res_traineduntouched.npy', resnet)
    """
}

ExtractProb = file('ExtractProbFromTrainedRes.py')

process ExtractProbFromTrained {
    clusterOptions "-S /bin/bash"
    queue "all.q"
    publishDir '../../partA/prob_from_pretrained_res', pattern: "*.tif", overwrite: true
    input:
    file py from ExtractProb
    file mean_file from MEAN
    file fold from IMAGE_FOLD
    file img from IMAGES
    file weights from Trained_files
    output:
    file '*.npy' into res_prob
    file "*_prob.tif"
    script:
    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/envs/cpu_tf/bin/python \$@
    }
    pyglib $py $img $weights $mean_file
    """
}

process RegroupProb {
    publishDir '../../partA/table', overwrite:true
    clusterOptions "-S /bin/bash -q all.q@compute-0-24"
    input:
    file tbls from res_prob .toList()
    output:
    file 'res_prob.npy' into RESPROB, RESPROB2
    script:
    """
    #!/usr/bin/env python
    import numpy as np
    from glob import glob
    files = glob('*.npy')
    size = np.load(files[0]).shape[0]
    resnet = np.zeros((len(files), size), dtype='float')
    for k, f in enumerate(files):
        resnet[k] = np.load(f)
    np.save('res_prob.npy', resnet)
    """
}

SUMMARIZE = file('Summarize_resnet.py')
FUNCTIONS = ["All", "Max", "Mean", "Median"]

process StatDescr {
    publishDir '../../partA/table', overwrite:true
    clusterOptions "-S /bin/bash -q all.q@compute-0-24"
    input:
    file table from RES2
    file py from SUMMARIZE
    each type from FUNCTIONS
    output:
    file "res_${type}.npy" into other_tabs
    script:
    """
    python $py $table $type res_${type}.npy
    """
}

RES.concat(other_tabs) .concat(RESTRAINED) .concat(RESPROB) .set{ALL_POS}

N_SPLIT = 5
TREE_SIZE = Channel.from([10, 100, 200, 500, 1000, 10000])
NUMBER_P = Channel.from(["auto", "log2"])
Channel.from(15..24) .concat(Channel.from(15..24)) .concat(Channel.from(15..24)) .concat(Channel.from(15..24)) .concat(Channel.from(15..24)) .set {COMP}
TREE_SIZE .combine(NUMBER_P) .merge(COMP).set{ Param }
ALL_POS .combine(Param) .set{ TAB_Param}

//TAB_Param .map{it -> file(it[0]).split()} .println()

process TrainRF {
    publishDir '../../partA/Results', overwrite: true
    clusterOptions "-S /bin/bash -q all.q@compute-0-${key}"
    input:
    val n_splits from N_SPLIT
    set file(table), n, method, key from TAB_Param
    output:
    file "score__${n}__${method}__${table.getBaseName().split('_')[1]}.csv" into RF_SCORES
    script:
    """
    #!/usr/bin/env python
    from sklearn.model_selection import StratifiedKFold
    from pandas import read_csv, DataFrame
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import numpy as np

    table_npy = np.load('${table}')
    id = table_npy[:,0].copy()
    y = table_npy[:,1].copy()
    res_y = np.zeros_like(y)
    X = table_npy[:,2:]
    skf = StratifiedKFold(n_splits=${n_splits}, shuffle=True, random_state=42)
    val_scores = np.zeros(${n_splits})
    cross = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=${n}, max_features='${method}')
        clf.fit(X_train, y_train)
        print 'Trained model for fold: {}'.format(cross)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        print 'Train Accuracy :: ', accuracy_score(y_train, y_pred_train)
        score_test = accuracy_score(y_test, y_pred_test)
        print 'Test Accuracy  :: ', score_test
        print ' Confusion matrix ', confusion_matrix(y_test, y_pred_test)
        res_y[test_index] = y_pred_test
        val_scores[cross] = score_test
        cross += 1
    DataFrame(val_scores).to_csv('score__${n}__${method}__${table.getBaseName().split('_')[1]}.csv')
    np.save('y_pred_${n}__${method}__${table.getBaseName().split('_')[1]}.npy', res_y)
    """
}

def getKey( file ) {
      file.name.split('__')[3] 
}
/* Regrouping files by patient for tiff stiching */
RF_SCORES  .map { file -> tuple(getKey(file), file) }
           .groupTuple() 
           .set { RES_BY_X }

process RegroupTables {
    publishDir '../../partA/Results', overwrite: true
    clusterOptions "-S /bin/bash"
    input:
    set key, file(_) from RES_BY_X 
    output:
    file "all_table_${key}" into FINAL_TABLE
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
        __, n_name, p_name, key = name.split('__')
        table.columns = ['n_{}_{}'.format(n_name, p_name)]
        l_tab.append(table)
    all_tab = concat(l_tab, axis=1)
    all_tab.ix['mean'] = all_tab.mean()
    all_tab.to_csv('all_table_${key}')
    """
}
