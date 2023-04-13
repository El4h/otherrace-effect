#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=12G   
#SBATCH --job-name='face'
#SBATCH --time=2-00:00:00
#SBATCH -p nklab
#SBATCH --mail-type=END
#SBATCH --mail-user=kdobs@mit.edu
#SBATCH --array=0-9
#SBATCH --output='./output/lesion_group/%A_%a.out'
#SBATCH --gres=gpu:1 # QUADRORTX6000:1 # GEFORCERTX2080TI:1 # gres=gpu:1
# #SBATCH --constraint=pascal|maxwell
    

# Usage: 
#      
#      SORT_TASK_INDEX=0
#      NONSORT_TASK_INDEX=1 
#      PARAM_GROUP_INDEX=0
#      sbatch lesion.sh $SORT_TASK_INDEX $NONSORT_TASK_INDEX $PARAM_GROUP_INDEX

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# CHANGE INDEXES HERE
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------



SORT_TASK_INDEX=1
NONSORT_TASK_INDEX=0
PARAM_GROUP_INDEX=12

#CONFIG_FILE='./configs/vgg/face_inanimate_400k_seed2.yaml'
CONFIG_FILE='./configs/vgg/face_dual_whitasia.yaml'
#CONFIG_FILE='./configs/vgg/face_inanimate_400k_seed.yaml'

# PARAM_WEIGHT_AND_BIAS_GROUP_INDEXES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) # for vgg16



GREEDY_P=0.25 # 0.2
GROUP_P=0.016 # 0.016
NGPUS=1
BATCH_SIZE=128
MAX_BATCHES=50
RESTORE_EPOCH=-1
LESION_NAME='2lesionlayer12' # must correspond to an already performe lesion
ITER_SEED_TYPE='selection'
READ_SEED=0
MAXOUT='True'
RAND_CLASSES='False'
RAND_CLASSES_SEED=1
SHUFFLE='False'

# # lesionEval.py
# DROP_PERCENTS_BEG=0
# DROP_PERCENTS_END=0.50
# DROP_PERCENTS_STEPSIZE=0.1
# EVAL_VERSION='eval1'
# ITERATOR_SEED=0
# PERF_MATCH=0.72 #0.72
# #SUBGROUPS_FILE='facedim.npy'


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

echo 'Terminal Command:'
echo '----------------------------------'
echo '$sbatch lesionGroup.sh '$SORT_TASK_INDEX' '$NONSORT_TASK_INDEX' '$PARAM_GROUP_INDEX
echo

echo 'Config File:'
echo '----------------------------------'
echo 'CONFIG_FILE:            '$CONFIG_FILE
echo
echo '-------INDEXING-------------------'
echo 'SLURM_ARRAY_TASK_ID:    '$SLURM_ARRAY_TASK_ID
echo
echo 'PARAM_GROUP_INDEX       '$PARAM_GROUP_INDEX
echo 'SORT_TASK_INDEX:        '$SORT_TASK_INDEX
echo 'NONSORT_TASK_INDEX:     '$NONSORT_TASK_INDEX
echo
echo '----------------------------------'

echo '-------PARAMS---------------------'
echo 'GREEDY_P:               '$GREEDY_P
echo 'GROUP_P:                '$GROUP_P
echo 'SHUFFLE:                '$SHUFFLE
echo 'NGPUS:                  '$NGPUS
echo 'BATCH_SIZE:             '$BATCH_SIZE
echo 'MAX_BATCHES:            '$MAX_BATCHES
echo 'RESTORE_EPOCH:          '$RESTORE_EPOCH
echo 'LESION_NAME:            '$LESION_NAME
echo 'EVAL_VERSION:           '$EVAL_VERSION
echo 'RAND_CLASSES:           '$RAND_CLASSES
echo 'RAND_CLASSES_SEED:      '$RAND_CLASSES_SEED
echo '----------------------------------'


echo 'submitting python script...'
echo


CUDA_VISIBLE_DEVICES=1 python lesionGroupFixingParser.py --config_file $CONFIG_FILE --param_group_index $PARAM_GROUP_INDEX --greedy_p $GREEDY_P --group_p $GROUP_P --shuffle $SHUFFLE --ngpus $NGPUS --batch_size $BATCH_SIZE --max_batches $MAX_BATCHES --sort_task_index $SORT_TASK_INDEX --nonsort_task_index $NONSORT_TASK_INDEX --restore_epoch $RESTORE_EPOCH --lesion_name $LESION_NAME --iterator_seed $ITER_SEED_TYPE --maxout $MAXOUT --read_seed $READ_SEED --randomize_classes $RAND_CLASSES --randomize_classes_seed $RAND_CLASSES_SEED >> output_$LESION_NAME.txt
# --subgroups_file $SUBGROUPS_FILE # -

#CUDA_VISIBLE_DEVICES=0 python lesionEval.py --config_file $CONFIG_FILE --param_group_index $PARAM_GROUP_INDEX --shuffle $SHUFFLE --ngpus $NGPUS --batch_size $BATCH_SIZE --maxout $MAXOUT --sort_task_index $SORT_TASK_INDEX --nonsort_task_index $NONSORT_TASK_INDEX --restore_epoch $RESTORE_EPOCH --lesion_name $LESION_NAME --eval_version $EVAL_VERSION --drop_percents_beg $DROP_PERCENTS_BEG --drop_percents_end $DROP_PERCENTS_END --drop_percents_stepsize $DROP_PERCENTS_STEPSIZE --iterator_seed $ITERATOR_SEED
# --random_lesion $RANDOM_LESION
