#!/bin/bash

# Range start, end, and increment
start=0
end=0
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
export prompt='Classify the satellite image. Provide only the class name as your answer.'

export dataset=eurosat
export data_path=/root/home/data/EuroSAT_RGB
export data_sample=/root/home/ViGIR_CVPR_LLM/data_split/split_zhou_EuroSAT.json
export subset=test
export results_dir=/root/home/ViGIR_CVPR_LLM/results/baselineExp/${dataset}
export timeout=40
export model_unloading=true  # Set as needed

#list_of_models=('llava:7b' 'llava:13b' 'llava:34b' 'llava-llama3' 'bakllava' 'moondream' 'minicpm-v' 'llava-phi3')
list_of_models=('llava:7b' 'llava:13b' 'llava:34b' 'llava-llama3' 'bakllava' 'minicpm-v' 'llava-phi3')



for (( modelno=start; modelno<=end; modelno+=increment )); do
  modelname=${list_of_models[${modelno}]}
  export modelno
  export modelname
  envsubst < job_nopvc_cvpr.yaml | kubectl apply -f -
done
