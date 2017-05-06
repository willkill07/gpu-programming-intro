#!/usr/bin/env bash
target=$(basename $1 .cpp)
shift

if [[ ! -e ${target}.cpp ]]
then
    exit
fi

for version in out/${target}/*
do
    tag=$(echo ${version} | cut -f2 -d.)
    echo $(basename ${version})
    PGI_ACC_NOTIFY=3 PGI_ACC_TIME=1 OMP_NUM_THREADS=6 ./${version} ${@}
done
