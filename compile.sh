#!/usr/bin/env bash
target=$1
shift

for version in Makefile.*
do
    tag=$(echo ${version} | cut -f2 -d.)
    make -f ${version} ${target}
    mv ${target} ${target}.${tag}
done
