dimensions=('subject_consistency' 'background_consistency' 'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality')
# dimension='background_consistency'
# dimensions=('imaging_quality' 'background_consistency')

for dimension in "${dimensions[@]}"; do
    CUDA_VISIBLE_DEVICES=3 vbench evaluate \
        --dimension $dimension \
        --videos_path outputs_eval/svd/baseline \
        --mode=custom_input \
        --output_path evaluation_results/svd/baseline/$dimension

    CUDA_VISIBLE_DEVICES=3 vbench evaluate \
        --dimension $dimension \
        --videos_path outputs_eval/svd/step2 \
        --mode=custom_input \
        --output_path evaluation_results/svd/step2/$dimension

    CUDA_VISIBLE_DEVICES=3 vbench evaluate \
        --dimension $dimension \
        --videos_path outputs_eval/svd/async2 \
        --mode=custom_input \
        --output_path evaluation_results/svd/async2/$dimension
done

# CUDA_VISIBLE_DEVICES=3 vbench evaluate \
#     --dimension $dimension \
#     --videos_path outputs_eval/svd/baseline \
#     --mode=custom_input \
#     --output_path evaluation_results/svd

# CUDA_VISIBLE_DEVICES=3 vbench evaluate \
#     --dimension $dimension \
#     --videos_path outputs_eval/svd/step2 \
#     --mode=custom_input \
#     --output_path evaluation_results/svd

# CUDA_VISIBLE_DEVICES=3 vbench evaluate \
#     --dimension $dimension \
#     --videos_path outputs_eval/svd/async2 \
#     --mode=custom_input \
#     --output_path evaluation_results/svd