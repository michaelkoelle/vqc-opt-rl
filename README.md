# A Study on Optimization Techniques for Variational Quantum Circuits in Reinforcement Learning

This GitHub repository includes the following files and directories:

- `jobs`:

  - `job.sh`
    - Required by `run-jobs.sh` and sets up a pyenv for testing.
  - `run-jobs.sh`
    - Executes slurm commands to run the algorithms on LRZ's computing resources.

- `plots`

  - Contains plots created with `plot_log.py` used in the paper.

- `qppo-slurm`

  - Checkpoints and results from all runs.

- `src`

  - Contains the Python files of the algorithm:
    - `agent.py` - Logic of the agent (Actor and Critic) and neural networks.
    - `args.py` - Passes hyperparameters.
    - `calc_num_param.py` - Calculates the number of parameters for the Actor and Critic for informational purposes.
    - `circuits.py` - Implementation of different circuits.
    - `env.setup.py` - Initialization function for individual environments for the multi-vector-environment.
    - `envs_storage.py` - Functions to store environment data for potential continuation of the learning process.
    - `layer_params.py` - Initialization of circuit parameters.
    - `main.py` - Executable file with the core logic of the (Quantum) Proximal Policy Optimization algorithm.
    - `plot_grads.py` - Functions for plotting gradients.
    - `plot_old.py` - Old plotting functions used for a simple final plot in `main.py`.
    - `plot.py` - Revised plotting functions used for evaluating test results.
    - `save_params.py` - Functions for saving circuit parameters (storing NN parameters in `agent.py`).
    - `save_results.py` - Functions for saving results.
    - `ShortestPathFrozenLake.py` - Modified Frozen Lake environment with dense reward.
    - `transform_funks.py` - Normalization and encoding functions for the circuit.
    - `utils.py` - Functions for determining the dimensions of the environment.

- `plot_log.py`

  - Executable file with all function calls used to create the plots.

- `run_log.txt`
  - List of all slurm commands used within `run-jobs.sh` to produce the test results.

## Running the Algorithm

To train the PPO in an OpenAI Gym environment (like Cart Pole), execute `$ python src/main.py --gym-id CartPole-v1`. To run multiple seeds in parallel via slurm, place multiple calls to `jobs/job.sh` (with the appropriate argparse arguments) in `jobs/run-jobs.sh` and start with `$ jobs/run-jobs.sh`.

## Hyperparameter Settings with argparse

A wide range of additional hyperparameters can be passed via the command line. These can be viewed with the `python src/main.py --help` command or in `src/args.py`. Since this can quickly become confusing, we have listed all the slurm-commands we executed (in `jobs/run-jobs.sh`) in `run_log.txt`.

## Requirements

The requirements listed in `requirements.txt` should be automatically installed when running `jobs/job.sh`. The algorithm has only been tested for Cart Pole and Frozen Lake and may require adjustments for other environments.

## Checkpoints

The algorithm saves all results and parameters at fixed intervals, which allows a run to be continued after the specified number of environment steps have passed or after a crash. For Frozen Lake, it was also possible to restore the state of the Multi Vector Environment. Unfortunately, for Cart Pole, we had to reinitialize the environments after each restoration. To load an existing checkpoint, the exact same function call must be executed again, but `--load-chkpt` must be set to `True` and a larger value can be used for `--total-timesteps`.

### Use of Checkpoints in our work

Although not perfect, the checkpoint system was indispensable for our tests in Cart Pole, as runs (using the Actor VQC) typically lasted about 3 days. Most runs in this environment were initially tested for 150,000 timesteps and then extended from the checkpoint to 500,000 steps. In case of node failures in the Slurm system, the process was also continued from the last checkpoint to avoid losing learning progress. The SPS (Steps per Second) plots should provide a good overview of when restarts occurred, as each restart results in a spike. Empirically, the impact of such a restart on the average performance of the (Quantum) PPO should be negligible, since the algorithm independently executes its learning cycles at fixed time intervals and only uses "On Policy" data (from the last learning interval), so no learning progress is lost. The only difference is that the random number generator is reset, and in the worst case, the final phase of a promising run (for each of the n parallel environments) cannot be observed. Given that over the course of 500,000 timesteps at least 1,000 (on average about 1,500 to 2,000) episodes are run, the impact of the reset should be relatively small. However, we wanted to mention this in terms of reproducibility.

## Plots

To plot the test results, run `src.plot.plot_test_avg_final("qppo-slurm/results", "plots", gym_id, exp_names, seeds, alpha, max_steps, labels)`, where:

- `gym_id` is the id of the Gym environment of the test.
- `exp_name` is a list of the names of the experiments to be plotted.
- `seeds` is a list of the seeds that were used.
- `alpha` is the alpha value for the EWM (Exponentially Weighted Moving Average).
- `max_steps` is the number of timesteps over which to plot.
- `labels` is a list of name abbreviations to be used for the legend (in the same order as `exp_name`).
  All the function calls we used are listed in `plot_log.py`.
