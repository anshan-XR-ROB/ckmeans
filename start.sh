#!/bin/bash

while [[ $# > 1 ]]; do
	key="$1"
	shift
	case $key in
		--feature_file_name)
		feature_file_name="$1"
		shift
		;;
	        --feature_num)
		feature_num="$1"
		shift
		;;
		--num_bits)
		num_bits="$1"
		shift
		;;
		--num_dim)
		num_dim="$1"
		shift
		;;
		--output_file_dir)
		output_file_dir="$1"
		shift
		;;
                --log_file)
                log_file="$1"
                shift
                ;;
                #*)
	        #echo 'unknown parameter '$1 # unknown option
		#exit -1
	        #;;
        esac
done
LD_LIBRARY_PATH=/opt/intel/composer_xe_2013_sp1.0.080/mkl/lib/intel64/ /home/anshan.as/ckmeans_train_platform/ckmeans_train $feature_file_name $feature_num $num_bits $num_dim $output_file_dir > $log_file &
pgrep ckmeans_train 
