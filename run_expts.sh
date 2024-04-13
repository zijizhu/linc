#! /bin/bash

set -e

outdir="outputs"
mkdir -p ${outdir}

for model in "google/gemma-7b"; do
    for base in "folio" "proofwriter"; do
        batch_size=5
        max_length=8192 # max model context including prompt
        for n in "8"; do
            for mode in "neurosymbolic"; do
                task="${base}-${mode}-${n}shot"
                run_id="${model#*/}_${task}"
                job="accelerate launch runner.py"
                job+=" --model ${model} --precision bf16"
                job+=" --tasks ${task} --n_samples 5 --batch_size ${batch_size}"
                job+=" --max_length_generation ${max_length} --temperature 0.8"
                job+=" --allow_code_execution --trust_remote_code --output_dir ${outdir}"
                job+=" --save_generations_raw --save_generations_raw_path ${run_id}_generations_raw.json"
                job+=" --save_generations_prc --save_generations_prc_path ${run_id}_generations_prc.json"
                job+=" --save_references --save_references_path ${run_id}_references.json"
                job+=" --save_results --save_results_path ${run_id}_results.json"
                job+=" &> ${outdir}/${run_id}.log; exit"
                # export JOB=${job}; bash SUBMIT.sh
                echo ${job}
                echo "Submitted ${run_id}"
            done
        done
    done
done
touch ${outdir}/run.done
