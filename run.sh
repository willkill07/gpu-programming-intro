#!/usr/bin/env bash
target=$1
shift

for version in Makefile.*
do
    # compile
    make -f ${version} ${target}
    # print header
    echo ${version}
    # run
    PGI_ACC_NOTIFY=31 PGI_ACC_TIME=1 OMP_NUM_THREADS=6 ./${target} ${@}
    #delete binary
    rm ${target}
done
