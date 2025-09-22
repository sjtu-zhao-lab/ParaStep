"""

"""

import torch
from audioldm_eval import EvaluationHelper

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

# stepparallelism_device2
# local_predict4
generation_result_path = "example/local_predict4"
target_audio_path = "example/baseline"

torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

# Initialize a helper instance
evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
    # backbone="cnn14", # `cnn14` refers to PANNs model, `mert` refers to MERT model
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)

print(metrics)