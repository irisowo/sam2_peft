export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_DISTRIBUTED_DEBUG=DETAIL


# Selection:
# LORA_r16, DORA_r16, RLRR, AdapterPlus_r8, ADA
# LORA_stage2_r16, DORA_stage2_r16, RLRR_stage2, ADA_stage2, 
pefttype='AdapterPlus_r8'
pefttype_stripped=${pefttype%%_r*}
pefttype_lower=$(echo "${pefttype_stripped}" | tr '[:upper:]' '[:lower:]')
config_file="configs/sam2.1_training/sam2.1_hiera_l_finetune_${pefttype_lower}.yaml"



export CUDA_VISIBLE_DEVICES="0"
export TMPDIR=/tmp
python training/train.py \
    -c ${config_file} \
    --use-cluster 0 \
    --num-gpus 1

# logfile="logs/train${pefttype}_sam2.1.log"
# errfile="logs/train${pefttype}_sam2.1.err"
# nohup python training/train.py \
#     -c ${config_file} \
#     --use-cluster 0 \
#     --num-gpus 1 > ${logfile} 2> ${errfile} &
