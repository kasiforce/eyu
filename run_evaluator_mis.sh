export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python ./run_src/do_eval.py \
    --dataset_name texthacker_humaneval_Mistral-7B-v0.1 \
    --model_ckpt mistralai/Mistral-7B-v0.1_2 \


#export TRANSFORMERS_OFFLINE=1
#export TOKENIZERS_PARALLELISM=false
#CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
#    --evaluator_device cuda:0 \
#    --evaluator_threshold 0.83 \
#    --mcts_num_last_votes 8 \
#    --dataset_name texthacker_humaneval_Mistral-7B-v0.1 \
#    --model_ckpt "mistralai/Mistral-7B-v0.1" \
#    --result_iteration 2 \
#    --disable_mutual_vote \
#    --disable_clone_detector \
#    --disable_gene_remain_subq \
#    --seed 88 \
#    --num_rollouts 16 \
#    --tensor_parallel_size 1 \
#    --gpu_memory_utilization 0.9 \
#    --verbose