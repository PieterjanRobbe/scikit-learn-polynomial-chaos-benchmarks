#! /bin/bash

for d in 2 4 8
do
    for k in 2 3 4 5 6 7
    do
        for m in openturns sklearn #pygpc chaospy 
        do
            echo "d = ${d}, k = ${k}, m = ${m}"
            python timing.py -d $d -k $k -m $m #&
        done
    done
done
#wait