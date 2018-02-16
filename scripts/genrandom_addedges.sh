#! /usr/bin/env bash

# Script to generate pooling instances using the "makereandom.py"
# script. Instances with 1 and 2 attributes are used. In the paper
# only the instances with 1 attribute are considered. The PhD thesis
# of Jonas Schweiger contains results for all instances.

# Call: genrandom_addedges.sh <destination folder>

# The output folder will be generated if not already present.

# List of number of additional attributes to generate instances for
ATTRIBUTES="0 1"

# The first argument is the output folder
FOLDER=$1

# make the output folder if not already present
mkdir -p $FOLDER

SEED=0

for i in $(seq 1 10);
do
    echo $i
    SIZE=10
    EDGES="10 20 30 40 50 60"

    for d in $EDGES;
    do
        for at in $ATTRIBUTES;
        do
            makerandom.py --seed $SEED --haverly $SIZE --scalehaverlys random --addedges $d --attributes $at ${FOLDER}/haverly_${SIZE}_addedges_${d}_attr_${at}_${i}
        done

        # increase the seed only here, so the seeds for the different numbers of attributes are the same
        let SEED+=1
    done

    SIZE=15
    EDGES="15 30 45 60 75 90"

    for d in $EDGES;
    do
        for at in $ATTRIBUTES;
        do
            makerandom.py --seed $SEED --haverly $SIZE --scalehaverlys random --addedges $d --attributes $at ${FOLDER}/haverly_${SIZE}_addedges_${d}_attr_${at}_${i}
        done

        # increase the seed only here, so the seeds for the different numbers of attributes are the same
        let SEED+=1
    done

    SIZE=20
    EDGES="20 40 60 80 100 120"

    for d in $EDGES;
    do
        for at in $ATTRIBUTES;
        do
            makerandom.py --seed $SEED --haverly $SIZE --scalehaverlys random --addedges $d --attributes $at ${FOLDER}/haverly_${SIZE}_addedges_${d}_attr_${at}_${i}
        done

        # increase the seed only here, so the seeds for the different numbers of attributes are the same
        let SEED+=1
    done


done
