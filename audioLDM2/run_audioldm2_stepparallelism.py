import torch
import torch.distributed as dist
import time
import argparse
import os
import scipy
from audioLDM2.stepParallelism_audioldm2 import AudioLDM2Pipeline, StepParallelism

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cvssp/audioldm2-large') 
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--islocal", type=str, default='False', help="set islocal to use BatchStep, or use ParaStep")
    parser.add_argument("--predict_len", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    local = False
    if args.islocal == 'True':
        local = True
    

    pipeline = AudioLDM2Pipeline.from_pretrained(args.model, torch_dtype=torch.float16, )

    _world_size = int(os.environ["WORLD_SIZE"])
    step_parallelism = StepParallelism(pipeline, _world_size, local=local, predict_len=args.predict_len, warm_up=args.warmup)

    prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
    negative_prompt = "Low quality."

    #warmup
    print("warmup")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    audio = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_length_in_s=10.0,
        num_waveforms_per_prompt=3,
        generator=generator,
    ).audios
    step_parallelism.clear_cache()


    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    start = time.time()
    audio = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_length_in_s=10.0,
        num_waveforms_per_prompt=3,
        generator=generator,
    ).audios
    step_parallelism.clear_cache()
    print(f"device{step_parallelism._rank} stepparallelism Time taken: {time.time()-start:.2f} seconds.")

    if step_parallelism._rank == 0:
        if local == False:
            scipy.io.wavfile.write("techno_device{}.wav".format(int(os.environ["WORLD_SIZE"])), rate=16000, data=audio[0])
        else:
            scipy.io.wavfile.write("techno_local_predict{}.wav".format(args.predict_len), rate=16000, data=audio[0])
    dist.destroy_process_group()