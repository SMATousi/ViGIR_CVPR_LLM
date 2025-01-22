#!/bin/bash

# Range start, end, and increment
start=1
end=7
increment=1


unset prompt
unset dataset
unset data_path
unset data_sample
unset subset
unset results_dir
unset timeout
unset model_unloading


# Echo each variable to confirm they are unset
echo "prompt: $prompt"
echo "dataset: $dataset"
echo "data_path: $data_path"
echo "data_sample: $data_sample"
echo "subset: $subset"
echo "results_dir: $results_dir"
echo "timeout: $timeout"
echo "model_unloading: $model_unloading"

# Define the prompt as a variable
export prompt='Given the texture image, classify it into one of the following categories: banded, blotchy, braided, bubbly, bumpy, chequered, cobwebbed, cracked, crosshatched, crystalline, dotted, fibrous, flecked, freckled, frilly, gauzy, grid, grooved, honeycombed, interlaced, knitted, lacelike, lined, marbled, matted, meshed, paisley, perforated, pitted, pleated, polka-dotted, porous, potholed, scaly, smeared, spiralled, sprinkled, stained, stratified, striped, studded, swirly, veined, waffled, woven, wrinkled, or zigzagged. Provide only the texture name as your answer.'

export dataset=dtd
export data_path=/root/home/data/DTD/images
export data_sample=/root/home/data/ViGIR_CVPR_LLM/data_split/split_zhou_DescribableTextures.json
export results_dir=/mnt/cvpr2025/results/baselineExp/${dataset}
export timeout=40
export model_unloading=true  # Set as needed

list_of_models=('llava:7b' 'llava:13b' 'llava:34b' 'llava-llama3' 'bakllava' 'moondream' 'minicpm-v' 'llava-phi3')

for (( modelno=start; modelno<=end; modelno+=increment )); do
  modelname=${list_of_models[${modelno}]}
  export modelno
  export modelname
  export dataset
  envsubst < A100-job-nopvc.yaml | kubectl apply -f -
done
