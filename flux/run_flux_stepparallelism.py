"""
# baseline
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node=1 --run-path flux/run_flux_stepparallelism.py
# parastep
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 --run-path flux/run_flux_stepparallelism.py
"""
import time
import argparse
import torch
import torch.distributed as dist
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
import os
import json
import tqdm

from flux.stepParallelism_flux import StepParallelism


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='black-forest-labs/FLUX.1-dev')   
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--warm_up", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--warm_gpu", type=str, default="True")
    parser.add_argument("--coco_eval", type=str, default="False")

    parser.add_argument("--dataset_size", type=int, default=1, help="size of dataset to be used for evaluation. Default is 50, max is 5000")
    parser.add_argument("--output", type=str, default='experiments/latency_and_performance/flux/cocoeval/outputs')
   
    args = parser.parse_args()

    coco_annotation_path = "/workspace/data/coco/annotations/captions_val2017.json"
    coco_images_path = "/workspace/data/coco/images/val2017"

    _world_size = int(os.environ["WORLD_SIZE"])
    _rank = int(os.environ["RANK"])

    warm_gpu = True if args.warm_gpu == "True" else False
    coco_eval = True if args.coco_eval == "True" else False
    use_baseline = False
    
    output_path = ""

    if use_baseline:
        output_path = os.path.join(args.output, f"baseline")
    else:
        output_path = os.path.join(args.output, f"step{_world_size}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    latency_output_path = os.path.join(output_path, "0latency.txt")

    torch.cuda.set_device(f"cuda:{_rank}")

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = T5EncoderModel.from_pretrained(
        args.model,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = FluxTransformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    pipeline = FluxPipeline.from_pretrained(
        args.model,
        text_encoder_2=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        local_files_only=True
        # device_map="balanced",
    )

    # pipeline.to('cuda')
    warm_up = args.warm_up

    step_parallelism = StepParallelism(pipeline, _world_size,args.warm_up, step_num=args.num_inference_steps)

    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()


    # warm up
    if warm_gpu == True:
        print("warm gpu ing")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        image = pipeline(args.prompt, guidance_scale=3.5, height=576, width=1024, num_inference_steps=args.num_inference_steps).images[0]
        dist.barrier()
        step_parallelism.clear_cache()
        print("warm gpu done")
    
    if coco_eval == False:
        # inference
        print("begin inference")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.cuda.synchronize()
        start = time.time()

        image = pipeline(args.prompt, guidance_scale=3.5, height=576, width=1024, num_inference_steps=args.num_inference_steps).images[0]
        torch.cuda.synchronize()
        print(f"device{_rank} Time taken: {time.time()-start:.2f} seconds.")
        if step_parallelism._rank == 0:
            if _world_size!=1:
                image.save(f"flux_worldsize{_world_size}_warm_up{args.warm_up}.png")
            else:
                image.save(f"flux_baseline.png")
    else:
        print("begin eval for coco dataset")
        times = []
        with open(coco_annotation_path, 'r') as coco_annotation:
            coco_annotation = json.load(coco_annotation)
            for i in tqdm.tqdm(range(args.dataset_size)):
                caption = coco_annotation['annotations'][i]['caption']
                imageid = coco_annotation['annotations'][i]['image_id']
                output_id = str(coco_annotation['annotations'][i]['id'])
                output_id = f"{str(output_id).zfill(12)}.jpg"
                # imagepath, example: 000000000139, 000000481404.jpg
                imageid = f"{str(imageid).zfill(12)}.jpg"
                image_path = os.path.join(coco_images_path, imageid)
                

                prompt = caption
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

                torch.cuda.synchronize()
                start = time.time()

                image = pipeline(prompt, guidance_scale=3.5, height=576, width=1024, num_inference_steps=args.num_inference_steps, generator = torch.Generator("cuda").manual_seed(args.seed)).images[0]

                step_parallelism.clear_cache()

                torch.cuda.synchronize()
                times.append(time.time()-start)

                if _rank == 0:
                    image_output_path = os.path.join(output_path, output_id)
                    print(f"image_output_path={image_output_path}")
                    image.save(image_output_path)
        
        print("Average time taken: ", sum(times)/len(times))
        with open(latency_output_path,"w") as f:
            f.write("Average time taken: {}".format(sum(times)/len(times)))



    
    
    dist.destroy_process_group()