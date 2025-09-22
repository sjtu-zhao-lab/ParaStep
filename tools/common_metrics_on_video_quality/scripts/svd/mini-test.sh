# baseline/step/async
# cocoeval_svd, 10个video
## test 0
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/baseline 

## baseline and step2, 97.18
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/step2 

## baseline and async2, 354.65
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/async2 

## baseline and step3, 233
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/step3

## baseline and async3, 619
python get_fvd_score.py --reference_path ../../outputs_eval/svd/baseline --input_path ../../outputs_eval/svd/async3 