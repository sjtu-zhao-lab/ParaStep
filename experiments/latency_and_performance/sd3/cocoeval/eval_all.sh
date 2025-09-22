source experiments/latency_and_performance/sd3/cocoeval/datasetlen.sh
mkdir experiments/latency_and_performance/sd3/cocoeval/outputs/eval_outputs

baseline="experiments/latency_and_performance/sd3/cocoeval/outputs/baseline/baseline"
script="tools/calculate_all_for_image.py"

echo "eval asyncdiff"
CUDA_VISIBLE_DEVICES=0 python ${script} --reference_dir ${baseline} --target_dir experiments/latency_and_performance/sd3/cocoeval/outputs/asyncdiff/2 > experiments/latency_and_performance/sd3/cocoeval/outputs/eval_outputs/asyncdiff_2.txt

echo "eval ParaStep"
CUDA_VISIBLE_DEVICES=0 python ${script} --reference_dir ${baseline} --target_dir experiments/latency_and_performance/sd3/cocoeval/outputs/stepparallelism/step2 > experiments/latency_and_performance/sd3/cocoeval/outputs/eval_outputs/stepparallelism_step2.txt

echo "eval PipeFusion"
CUDA_VISIBLE_DEVICES=0 python ${script} --reference_dir ${baseline} --target_dir experiments/latency_and_performance/sd3/cocoeval/outputs/xdit/ringNone_pp2 > experiments/latency_and_performance/sd3/cocoeval/outputs/eval_outputs/xdit_pipe2.txt