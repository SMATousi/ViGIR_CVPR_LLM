#!/bin/bash

# Range start, end, and increment
start=1
end=7
increment=1

#list_of_models=('llava:7b' 'llava:13b' 'llava:34b' 'llava-llama3' 'bakllava' 'moondream' 'minicpm-v' 'llava-phi3')
list_of_models=('llava:7b' 'llava:13b' 'llava:34b' 'llava-llama3' 'bakllava' 'minicpm-v' 'llava-phi3')

for (( modelno=start; modelno<=end; modelno+=increment )); do
  modelname=${list_of_models[${modelno}]}
  job_name="inference-job-${modelno}-nopvc-eurosat"
  kubectl delete job $job_name
done
