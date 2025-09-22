source experiments/latency_and_performance/sd3/cocoeval/datasetlen.sh
echo datasetlen=${datasetlen}

SOURCE_DIR=$(pwd)

echo "begin generating for baseline"
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 --master_port=12701 --run-path experiments/latency_and_performance/sd3/cocoeval/cocoeval.py --dataset_size ${datasetlen} --output "experiments/latency_and_performance/sd3/cocoeval/outputs/baseline" 

echo "begin generating for ParaStep"
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node=2 --run-path experiments/latency_and_performance/sd3/cocoeval/cocoeval.py --baseline False --dataset_size ${datasetlen} --output "experiments/latency_and_performance/sd3/cocoeval/outputs/stepparallelism"


echo "begin generating for Asyncdiff"
cd baseline/AsyncDiff 
echo "now_path is $(pwd)"
CUDA_VISIBLE_DEVICES=1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m torch.distributed.run --nproc_per_node=2 --run-path cocoeval.py --model_n 2 --stride 1 --dataset_size ${datasetlen} --output "/workspace/projects/diffusion/ParaStep-anonymous/experiments/latency_and_performance/sd3/cocoeval/outputs/asyncdiff/2"


cd ${SOURCE_DIR}
echo "begin generating for PipeFusion"
echo "now_path is $(pwd)"
CUDA_VISIBLE_DEVICES=1,2 datasetlen=${datasetlen} torchrun --nproc_per_node=2 \
baseline/xDiT/cocoeval.py \
--model "stabilityai/stable-diffusion-3-medium-diffusers" \
--num_inference_steps 50 \
--seed 20 \
--prompt "A cat holding a sign that says hello world" \
--height 1440 \
--width 1440  \
--pipefusion_parallel_degree 2 \
--warmup_steps 5