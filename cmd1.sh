#!/bin/bash
seed=0

#data="synthetic"
#experiment="synthetic"
#for batch_effect in "lv1" "lv3"
#do
#	for mode in "SRPS" "SRPS(no_baseline)" "SRPS(soft)" "deepCLife" "SourceOnly" "DANN" "RandomForestHarmony" "RandomForest"
#	do
#		for ((para_id=0; para_id<128; para_id++))
#		do
#			for ((j=0; j<5; j++))
#			do
#				python3.7 scripts_py/train_models.py --data $data --experiment $experiment --batch_effect $batch_effect --seed $seed --test_fold $j --mode $mode --para_id $para_id --device /cpu:0
#			done
#		done
#	done
#done
#
# data="HCC"
# experiment="Jiang2Gao"
# for mode in "SRPS" "deepCLife" "SourceOnly" "DANN" "RandomForestHarmony" "RandomForest"
# do
# 	for ((para_id=0; para_id<128; para_id++))
# 	do
# 		for ((j=0; j<5; j++))
# 		do
# 			python3.7 scripts_py/train_models.py --data $data --experiment $experiment --seed $seed --test_fold $j --mode $mode --para_id $para_id --device /cpu:0
# 		done
# 	done
# done

data="HCC_LUAD"
experiment="Jiang2Xu"
for mode in "SRPS" "deepCLife" "SourceOnly" "DANN" "RandomForestHarmony" "RandomForest"
do
	for ((para_id=0; para_id<1; para_id++))
	do
		for ((j=0; j<5; j++))
		do
			python3.7 scripts_py/train_models.py --data $data --experiment $experiment --seed $seed --test_fold $j --mode $mode --para_id $para_id --device /cpu:0
		done
	done
done

#data="toy"
#experiment="toy"
#for ((os_i=0; os_i<11; os_i++))
#do
#	for ((noise_j=0; noise_j<11; noise_j++))
#	do
#		for mode in "SRPS" "SourceOnly"
#		do
#			python3.7 scripts_py/train_models.py --data $data --experiment $experiment --seed $seed --mode $mode --para_id 0 --os_swap_i $os_i --batch_noise_j $noise_j --device /cpu:0
#		done
#	done
#done
