#! /bin/bash

for d in $1/*/ ; do
    #python3 exportTensorFlowLog.py $d $2/$d scalars
    for sub in $d*/ ; do
        python3 exportTensorFlowLog.py $sub $2/$sub scalars
    done
done