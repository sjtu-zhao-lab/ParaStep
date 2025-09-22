import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserStableDiffusion3Pipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import get_data_parallel_world_size
import json
import tqdm
import torch.distributed as dist

def main():
    coco_annotation_path = "datasets/coco/captions_val2017.json"
    dataset_size = int(os.environ['datasetlen'])
    print(f"dataset_size=={dataset_size}")
    # _rank = int(os.environ["RANK"])
    output = os.environ.get('output', "experiments/latency_and_performance/sd3/cocoeval/outputs/xdit")  # 默认值为100
    print(f"output=={output}")

    # output = "experiments/latency_and_performance/sd3/cocoeval/outputs/xdit"

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    text_encoder_3 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_3", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        print(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_3, weights=qfloat8)
        freeze(text_encoder_3)

    pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        text_encoder_3=text_encoder_3,
    ).to(f"cuda:{local_rank}")
    print("warm gpu") 
    pipe.prepare_run(input_config)
    _ = pipe(
        negative_prompt="",
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        guidance_scale=7.0,
        generator = torch.Generator("cuda").manual_seed(args.seed)
    )



    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    outputpath = f"ring{engine_args.ring_degree}_pp{engine_args.pipefusion_parallel_degree}"
    outputpath = os.path.join(output, outputpath)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    latency_output_path = os.path.join(outputpath, "0latency.txt")
    times = []
    with open(coco_annotation_path, 'r') as coco_annotation:
        coco_annotation = json.load(coco_annotation)
        for i in tqdm.tqdm(range(dataset_size)):
            caption = coco_annotation['annotations'][i]['caption']
            imageid = coco_annotation['annotations'][i]['image_id']
            output_id = str(coco_annotation['annotations'][i]['id'])
            output_id = f"{str(output_id).zfill(12)}.jpg"
            # imagepath, example: 000000000139, 000000481404.jpg
            imageid = f"{str(imageid).zfill(12)}.jpg"
            # image_path = os.path.join(coco_images_path, imageid)
            
            prompt = caption

            input_config.prompt = prompt

            # pipe.prepare_run(input_config)

            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            pipe.prepare_run(input_config)
            output = pipe(
                negative_prompt="",
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                guidance_scale=7.0,
                generator = torch.Generator("cuda").manual_seed(args.seed)
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

        

            parallel_info = (
                f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
                f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
                f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
            )
            if input_config.output_type == "pil":
                dp_group_index = get_data_parallel_rank()
                num_dp_groups = get_data_parallel_world_size()
                dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
                if pipe.is_dp_last_group():
                    if not os.path.exists("results"):
                        os.mkdir("results")
                    for i, image in enumerate(output.images):
                        image_rank = dp_group_index * dp_batch_size + i
                        if True:
                            image_output_path = os.path.join(outputpath, output_id)
                            print(f"image_output_path={image_output_path}")
                            image.save(
                                image_output_path
                            )


    
    with open(latency_output_path,"w") as f:
        f.write("Average time taken: {}".format(sum(times)/len(times)))
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
