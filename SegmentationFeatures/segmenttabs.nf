

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
