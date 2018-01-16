
IMAGE_FOLD = file('/share/data40T_v2/Peter/ICIAR2018/partB/WSI')
NUMBER = Channel.from(1..10)
SAMPLES_PER_CLASS = 1000
CUTWSI = file('cutWSI.py')


process cutWSI {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB/", pattern:"A*.png"
    queue 'all.q'
    input:
    file fold from IMAGE_FOLD
    val num from NUMBER
    val samples from SAMPLES_PER_CLASS
    file py from CUTWSI

    output:
    file "samples/*.png" into INPUT_DL mode flatten
    file "patch_extract/*.png"

    """
    python $py $fold $num $samples
    """
}

THRESH = 0.5
REMOVEWHITEPICS = file('removeWhitePics.py')

process removeWhitePics {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB/samples"
    queue 'all.q'
    input:
    file py from REMOVEWHITEPICS
    file _ from INPUT_DL 
    val thresh from THRESH
    output:
    file "valid/*.png" into INPUT_VALID mode flatten
    file "discard/*.png" 
    """
    python $py $thresh
    """
}




SPLIT = 5
EPOCH = 10
BATCH = 32
RESNET_50 = file('resnet_50.py')
PRETRAINED = file('imagenet_models')
LEARNING_RATE= [0.01, 0.001, 0.0001]
MOMENTUM = [0.5, 0.9, 0.99]
WEIGHT_DECAY = [0.0005, 0.00005, 0.000005]

process deepTrain {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB/ResultTest"
    queue 'cuda.q'
    maxForks 1
    input:
    file py from RESNET_50
    val split from SPLIT
    val epoch from EPOCH
    val batch_size from BATCH
    file _ from PRETRAINED
    file images from INPUT_DL .toList()
    each lr from LEARNING_RATE
    each mom from MOMENTUM
    each w_d from WEIGHT_DECAY
    output:
    file "${lr}__${mom}__${w_d}.csv" into RES_TRAIN

    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    pyglib $py --split $split --epoch $epoch --bs $batch_size --lr $lr --mom $mom --weight_decay ${w_d} --output ${lr}__${mom}__${w_d}.csv
    """
}
