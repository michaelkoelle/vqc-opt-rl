#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name5="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-4params-1.25e-2-(76-params)"
name6="FL-qppo-ac-simple_reuploading-qlr-2.5e-3------------------output_scaleing-4params-1.25e-2-(76-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=50000

circuits=("simple_reuploading")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh  --exp-name $name5  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1.25e-2 --shared-output-scaleing-param False
            sbatch --job-name="run-$env-$name6-$seed" jobs/job.sh  --exp-name $name6  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1.25e-2 --shared-output-scaleing-param False
        done
    done
done
