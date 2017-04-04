#!/bin/bash

input="Random"
output=$input"_Crop/"
input=$input"/"	
i=1
n=`ls $input | wc -l`
n=$(($n-1))

for file in $input*.jpg; do
    out="$file ($i/$n)"
    l=${#out}
    s=$(printf "%-${l}s" " ")
    echo -en "\e[0K\r$out"
    python2 photo_crop.py $file $output
    i=$(($i+1))
    if [ $i -lt $n ]; then
	echo -en "\e[0K\r$s"
    fi
done

echo " "
