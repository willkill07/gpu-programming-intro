#!/usr/bin/env bash
target=$(basename $1 .cpp)
shift

if [[ ! -e ${target}.cpp ]]
then
    exit
fi

mkdir -p out/${target}
for version in build/Makefile.*
do
    tag=$(echo ${version} | cut -f2 -d.)
    make -f ${version} ${target}
    mv ${target} out/${target}/${tag}
done
