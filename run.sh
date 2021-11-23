#!/bin/bash

for py_file in $(find '/home/pierre/Documents/PHD/fsfcos/run_exp' -name '*.py')
do
    bn="$(basename $py_file)"
    script="${bn%%.*}"
    reject="__init__"
    echo $script
    if [ "$script" != "$reject" ]; then
    
    	python -m "fsfcos.run_exp.$script"
    else
    	echo "Wrong python file, continue."
    fi
done
echo "Training complete!"
