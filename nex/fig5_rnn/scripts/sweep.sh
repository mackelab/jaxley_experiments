#! /bin/bash

# $1 amount of jobs
# $2 sweep id

#python_path=
#train_script=
#save_dir=

# Remove the existing temp logs and make the directory again
rm -rf $save_dir
mkdir $save_dir

for ((i=0; i<$1; ++i)); do
    nohup $python_path -u $train_script --sweep_id $2 > $save_dir$2_$i.out &
    echo $! >> $save_dir/save_pid.txt
    sleep 5
done