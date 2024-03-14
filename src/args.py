import argparse
import json
import os
from distutils.util import strtobool


def save_args(args):
    args_chkpt_file = os.path.join(args.chkpt_dir, "commandline_args.txt")
    with open(args_chkpt_file, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of the experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="CartPole-v1",
        help="id of the OpenAi Gym Environment /default: CartPole-v1",
    )
    parser.add_argument(
        "--exp-scheduling-qlearning-rate",
        type=float,
        default=10e-3,
        help="learning rate at start of exp-scheduling, will exponentially decline to the normal q-lr depending on the exp-scheduling-halftime",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="classic critic NN optimizer learning rate",
    )
    parser.add_argument(
        "--classic-actor-learning-rate",
        type=float,
        default=2.5e-4,
        help="classic actor NN optimizer learning rate",
    )
    parser.add_argument(
        "--qlearning-rate",
        type=float,
        default=1e-3,
        help="quantum circuit optimizer learning rate",
    )
    parser.add_argument(
        "--output-scaleing-learning-rate",
        type=float,
        default=1e-3,
        help="output scaleing learning rate",
    )
    parser.add_argument(
        "--input-scaleing-learning-rate",
        type=float,
        default=1e-4,
        help="input scaleing learning rate, currently onely needed for shared (global) input scaleing, since it is set to the current qlr else",
    )
    parser.add_argument(
        "--output-scaleing-start", type=float, default=1.0, help="output scaleing start value"
    )
    parser.add_argument("--seed", type=int, default=1, help="seed for all random events")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="lenght of the Training (number of timesteps in the Environment)",
    )
    parser.add_argument(
        "--exp-scheduling-halftime",
        type=int,
        default=150000,
        help="length (number of timesteps) of the exp-scheduling half time in which the exp-scheduling-qlearning-rate halves",
    )

    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="set cdnn.deterministic /default:True",
    )

    # Algorith specific
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="number of environments that run in parrallel in the SyncVectorEnv",
    )  # 1 Vector environment
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps that are run in each environments for each update step",
    )
    parser.add_argument(
        "--exp-qlr-scheduling",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="exponential learning rate scheduling for quantum circuits, overrides other scedulings",
    )

    parser.add_argument(
        "--sigmoid-qlr-scheduling",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="sigmoid learning rate scheduling for quantum circuits, overrides other scedulings",
    )

    
    parser.add_argument(
        "--arctan-input-rescale",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="uses manual 2*arctan(x) input rescaleing for cartpole environment (if no trainable input rescaleing is activated)",
    )
    

    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="select GAE for Advantage computation",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor gamma")  # 0.98
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="lambda for general advantage estimation if gae is active",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="number of minibatches the batch is devided into",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="in every update phase the algorithm devides the batch into minibatches X times to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle Advantage normalisation",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="clipping coefficient for ratio and v_loss clipping",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle the use of clipped value loss",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="multiplicative weight for the entropy (loss), 0 = disabled",
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="multiplicative weight for the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="maximum norm for the global gradient clipping",
    )

    parser.add_argument(
        "--actor-hidden-layer1-nodes",
        type=int,
        default=64,
        help="number of nodes for the first hidden layer of the actor",
    )
    parser.add_argument(
        "--actor-hidden-layer2-nodes",
        type=int,
        default=64,
        help="number of nodes for all second hidden layer of the actor, if set to 0 onely one hidden layer will be used",
    )
    parser.add_argument(
        "--critic-hidden-layer-nodes",
        type=int,
        default=64,
        help="number of nodes for all hidden layers of the critic",
    )
    parser.add_argument(
        "--quantum-actor",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if True a VQC will be used as actor",
    )
    parser.add_argument(
        "--quantum-critic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if True a VQC will be used as critic",
    )
    parser.add_argument("--n-qubits", type=int, default=4, help="number of qubits of the circuit")
    parser.add_argument(
        "--n-var-layers",
        type=int,
        default=2,
        help="gives the number of variational layers",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=1,
        help="gives the the number of encodeing layers (currently onely used for calculation of the number of parameters for info, in circuits the number of enc layers is given by the number of var layers)",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="simple",
        help="the circuit that is to be used, can be one of:  simple / simple_reuploading / simple_reuploading_with_shared_input_scaleing / simple_reuploading_with_input_scaleing / Hgog / Hgog_reuploading / Hgog_reuploading_with_input_scaleing / Jerbi-no-reuploading-no-input-scaleing / Jerbi-reuploading-no-input-scaleing / Jerbi-reuploading",
    )
    parser.add_argument(
        "--output-scaleing",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle output scaleing",
    )
    parser.add_argument(
        "--shared-output-scaleing-param",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if true onely one parameter is used for output scaleing, else one for every action",
    )
    parser.add_argument(
        "--hybrid",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle hybrid output postprocessing",
    )
    parser.add_argument(
        "--load-chkpt",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="continue learning from checkpoint (a checkpoint wizh the same exp_name has to exist for this to work, stored args are used exept total-timesteps)",
    )
    parser.add_argument(
        "--save-location",
        type=str,
        default="qppo-slurm",
        help="save-location/gym-id/exp-name/seed/ will make the save location of the experiment checkpoint, data and plots",
    )
    parser.add_argument(
        "--save-intervall",
        type=int,
        default=20,
        help="gives the the checkpoint save-intervall in number of updates (batchsize = num_steps * num_envs)",
    )
    parser.add_argument(
        "--random-baseline",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle completely random action selection",
    )

    parser.add_argument(
        "--clip-circuit-grad-norm",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle clipping of circuit gradient norm",
    )

    parser.add_argument(
        "--record-grads",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="toggle recording of gradient mean and std",
    )

    parser.add_argument(
        "--param-init",
        type=str,
        default="random",
        help="the parameter initialisation that is to be used, can be one of:  random / random_clipped / unclipped_gauss_distribution / gauss_distribution / rescaled_gauss_distribution / allverysmall / allsmall / allmid / allbig",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_steps * args.num_envs)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.chkpt_dir = f"{args.save_location}/checkpoints/{args.gym_id}/{args.exp_name}/{args.seed}"
    args.results_dir = f"{args.save_location}/results/{args.gym_id}/{args.exp_name}/{args.seed}"
    args.plot_dir = f"{args.save_location}/plots/{args.gym_id}/{args.exp_name}/{args.seed}"

    if (
        args.load_chkpt
    ):  # load args exept total_timesteps and continue learning if new total_timesteps > global_timesteps
        args_chkpt_file = os.path.join(args.chkpt_dir, "commandline_args.txt")
        t = args.total_timesteps
        with open(args_chkpt_file, "r") as f:
            args.__dict__ = json.load(f)
        args.load_chkpt = True
        args.total_timesteps = t

    return args
