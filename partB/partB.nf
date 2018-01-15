
IMAGE_FOLD = file('/share/data40T_v2/Peter/ICIAR2018/partB/WSI')
NUMBER = Channel.from(1..10)
SAMPLES_PER_CLASS = 1000
CUTWSI = file('cutWSI.py')


process cutWSI {
    clusterOptions "-S /bin/bash"
    publishDir "../../partB/"
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

SPLIT = 5
EPOCH = 10
BATCH = 32
RESNET_50 = file('resnet_50.py')

process deepTrain {
    clusterOptions "-S /bin/bash"
    publishDir "../partB/"
    input:
    file py from RESNET_50
    val split from SPLIT
    val epoch from EPOCH
    val batch_size from BATCH
    file images from INPUT_DL .toList()
    output:

    """
    function pyglib {
        /share/apps/glibc-2.20/lib/ld-linux-x86-64.so.2 --library-path /share/apps/glibc-2.20/lib:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/cuda/lib64/:/cbio/donnees/pnaylor/cuda/lib64:/usr/lib64/nvidia /cbio/donnees/pnaylor/anaconda2/bin/python \$@
    }
    pyglib $py $split $epoch $batch_size
    """
}
