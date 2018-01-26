
IMAGE_FOLD = file('/data/users/pnaylor/ICIAR2018/partB/WSI')
NUMBER = Channel.from(1..10)
SAMPLES_PER_CLASS = 1000
CUTWSI = file('cutWSI.py')


process cutWSI {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB_2/", pattern:"A*.png", overwrite:true
    queue 'all.q'
    input:
    file fold from IMAGE_FOLD
    val num from NUMBER
    val samples from SAMPLES_PER_CLASS
    file py from CUTWSI

    output:
    file "samples/*.png" into INPUT_DL
    file "patch_extract/*.png"

    """
    python $py $fold $num $samples
    """
}

THRESH = 0.7
REMOVEWHITEPICS = file('removeWhitePics.py')

process removeWhitePics {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB_2/samples", overwrite:true
    queue 'all.q'
    input:
    file py from REMOVEWHITEPICS
    file _ from INPUT_DL 
    val thresh from THRESH
    output:
    file "valid_*" into INPUT_VALID, INPUT_VALID2 mode flatten
    file "discard_*" 
    """
    python $py $thresh
    """
}

process mean {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB_2/meta", overwrite:true
    queue 'all.q'
    input:
    file _ from INPUT_VALID .toList()
    output:
    file "mean.npy" into MEAN_ARRAY
    """
    #!/usr/bin/env python
    import numpy as np
    from skimage.io import imread
    from glob import glob
    files = glob('valid_*/*.png')
    mean_ = np.zeros(3, dtype='float')
    for k, f in enumerate(files):
        mean_ += np.mean(imread(f)[:,:,0:3], axis=(0,1))
    mean_ = mean_ / len(files)
    np.save('mean.npy', mean_)
    """

}


SPLIT = 10
EPOCH = 10
BATCH = 64
RESNET_50 = file('resnet_50.py')
PRETRAINED = file('imagenet_models')
//LEARNING_RATE= [0.0001, 0.00001]
LEARNING_RATE= [0.01, 0.001, 0.0001, 0.00001]
MOMENTUM = [0.9, 0.99]
WEIGHT_DECAY = [0.00005]
process deepTrain {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB_2/ResultTest", overwrite:true
    queue 'cuda.q'
    maxForks 1
    beforeScript "source /data/users/pnaylor/CUDA_LOCK/.whichNODE"
    afterScript "source /data/users/pnaylor/CUDA_LOCK/.freeNODE"
    input:
    file mean from MEAN_ARRAY
    file py from RESNET_50
    val split from SPLIT
    val epoch from EPOCH
    val batch_size from BATCH
    file _ from PRETRAINED
    file images from INPUT_VALID2 .toList()
    each lr from LEARNING_RATE
//    each mom from MOMENTUM
//    each w_d from WEIGHT_DECAY
    output:
    file "${lr}__adam.csv" into RES_TRAIN
    file "*.h5" into MODEL_WEIGHTS

    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    python $py --split $split --epoch $epoch --bs $batch_size --lr $lr --output ${lr}__adam.csv --output_mod ${lr}__adam.h5 --mean $mean
    """
}

PredWSI = file('PredWSI.py')

process PrepareWSI {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB_2/SampleCheckForStep2", overwrite:true
    queue 'cuda.q'
    input:
    file mean from MEAN_ARRAY
    file py from PredWSI
    file fold from IMAGE_FOLD
    file weights from MODEL_WEIGHTS
    output:
    file "*.png" into STEP2INPUT

    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    python $py --mean $mean --fold $fold 
    """
}
