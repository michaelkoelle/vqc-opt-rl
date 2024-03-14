from args import parse_args
from utils import get_act_dim, get_obs_dim

args = parse_args()

# number of parameters for a neural network layer: number_of_inputs * number_of_outputs + number_of_outputs
#                                                 (             weights              ) + (     biases    )

# number of parameters for a quantum circuit:      qubits * layers * dimensions   +   (output scaleing   or   hybrid neural network)


# calculate number of trainable actor parameters
###############################################################
def calc_num_actor_params(envs):
    actor_par_count = 0
    """
    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
    ):
        num_obs = 16
        num_acts = 4
        debug = 1
    elif args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
        num_obs = 4
        num_acts = 2
        debug = 1
    else:
        num_obs = 0
        num_acts = 0
        debug = 0
        print("number of actor parameters calculation error")
    """
    num_obs = get_obs_dim(envs.single_observation_space)
    num_acts = get_act_dim(envs.single_action_space)

    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        actor_par_count = 3 * args.n_var_layers * args.n_qubits
    elif args.quantum_actor and (args.circuit == "simple_reuploading_with_shared_input_scaleing"):
        actor_par_count = 3 * args.n_var_layers * args.n_qubits + 1 * args.n_qubits
    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        actor_par_count = (
            3 * args.n_var_layers * args.n_qubits + 1 * args.n_enc_layers * args.n_qubits
        )
    elif args.quantum_actor and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        actor_par_count = 2 * (args.n_enc_layers + 1) * args.n_qubits
    elif args.quantum_actor and args.circuit == "Jerbi-reuploading":
        actor_par_count = 2 * (2 * args.n_enc_layers + 1) * args.n_qubits
    elif not args.quantum_actor:
        if args.actor_hidden_layer2_nodes != 0:  # 2 hidden layers
            actor_par_count = (
                num_obs * args.actor_hidden_layer1_nodes  # weights
                + args.actor_hidden_layer1_nodes * args.actor_hidden_layer2_nodes  # weights
                + args.actor_hidden_layer2_nodes * num_acts  # weights
                + args.actor_hidden_layer1_nodes  # bias
                + args.actor_hidden_layer2_nodes  # bias
                + num_acts  # bias
            )
        else:  # 1 hidden layer
            actor_par_count = (
                num_obs * args.actor_hidden_layer1_nodes  # weights
                + args.actor_hidden_layer1_nodes * num_acts  # weights
                + args.actor_hidden_layer1_nodes  # bias
                + num_acts  # bias
            )

    if args.quantum_actor and args.hybrid:
        actor_par_count = actor_par_count + args.n_qubits * num_acts + num_acts
    elif (
        args.quantum_actor
        and not args.hybrid
        and args.output_scaleing
        # and not args.scheduled_output_scaleing
    ):
        if not args.shared_output_scaleing_param:
            actor_par_count = actor_par_count + num_acts
        else:
            actor_par_count = actor_par_count + 1
    return actor_par_count


###############################################################


# calculate number of trainable critic parameters
###############################################################
def calc_num_critic_params(envs):
    critic_par_count = 0
    """
    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
    ):
        num_obs = 16
        debug = 1
    elif args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
        num_obs = 4
        debug = 1
    else:
        num_obs = 0
        debug = 0
        print("number of critic parameters calculation error")
    """
    num_obs = get_obs_dim(envs.single_observation_space)

    if args.quantum_critic and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        critic_par_count = 3 * args.n_var_layers * args.n_qubits
    elif args.quantum_critic and (args.circuit == "simple_reuploading_with_shared_input_scaleing"):
        critic_par_count = 3 * args.n_var_layers * args.n_qubits + 1 * args.n_qubits
    elif args.quantum_critic and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        critic_par_count = (
            3 * args.n_var_layers * args.n_qubits + 1 * args.n_enc_layers * args.n_qubits
        )
    elif args.quantum_critic and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        actor_par_count = 2 * (args.n_enc_layers + 1) * args.n_qubits
    elif args.quantum_critic and args.circuit == "Jerbi-reuploading":
        critic_par_count = 2 * (2 * args.n_enc_layers + 1) * args.n_qubits
    elif not args.quantum_critic:
        critic_par_count = (
            num_obs * args.critic_hidden_layer_nodes
            + args.critic_hidden_layer_nodes**2
            + args.critic_hidden_layer_nodes * 1
            + 2 * args.critic_hidden_layer_nodes
            + 1
        )

    if args.quantum_critic and args.hybrid:
        critic_par_count = critic_par_count + args.n_qubits * 1 + 1
    return critic_par_count


###############################################################


def manually_calc_num_params(
    gym_id="CartPole-v1",
    actor=True,
    quantum=False,
    circuit="NN",
    hidden_layer_nodes1=4,
    hidden_layer_nodes2=4,
    n_var_layers=6,
    n_enc_layers=6,
    n_qubits=4,
    hybrid=False,
    output_scaleing=False,
    # scheduled_output_scaleing=False,
    shared_output_scaleing_param=True,
):
    actor_par_count = 0
    if (
        gym_id == "FrozenLake-v0"
        or gym_id == "FrozenLake-v1"
        or gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
    ):
        num_obs = 16
        if actor:
            num_acts = 4
        else:
            num_acts = 1
    elif gym_id == "CartPole-v0" or gym_id == "CartPole-v1":
        num_obs = 4
        if actor:
            num_acts = 2
        else:
            num_acts = 1
    else:
        num_obs = 0
        num_acts = 0
        print("number of actor parameters calculation error")

    if quantum and (
        circuit == "simple"
        or circuit == "Hgog"
        or circuit == "simple_reuploading"
        or circuit == "Hgog_reuploading"
    ):
        actor_par_count = 3 * n_var_layers * n_qubits
    elif quantum and (circuit == "simple_reuploading_with_shared_input_scaleing"):
        actor_par_count = 3 * n_var_layers * n_qubits + 1 * n_qubits
    elif quantum and (
        circuit == "simple_reuploading_with_input_scaleing"
        or circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        actor_par_count = 3 * n_var_layers * n_qubits + 1 * n_enc_layers * n_qubits
    elif quantum and (
        circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        actor_par_count = 2 * (n_enc_layers + 1) * n_qubits
    elif quantum and circuit == "Jerbi-reuploading":
        actor_par_count = 2 * (2 * n_enc_layers + 1) * n_qubits
    elif not quantum:
        if hidden_layer_nodes2 != 0:  # 2 hidden layers
            actor_par_count = (
                num_obs * hidden_layer_nodes1  # weights
                + hidden_layer_nodes1 * hidden_layer_nodes2  # weights
                + hidden_layer_nodes2 * num_acts  # weights
                + hidden_layer_nodes1  # bias
                + hidden_layer_nodes2  # bias
                + num_acts  # bias
            )
        else:  # 1 hidden layer
            actor_par_count = (
                num_obs * hidden_layer_nodes1  # weights
                + hidden_layer_nodes1 * num_acts  # weights
                + hidden_layer_nodes1  # bias
                + num_acts  # bias
            )

    if quantum and hybrid:
        actor_par_count = actor_par_count + n_qubits * num_acts + num_acts
    elif quantum and not hybrid and output_scaleing:  # and not scheduled_output_scaleing
        if not shared_output_scaleing_param:
            actor_par_count = actor_par_count + num_acts
        else:
            actor_par_count = actor_par_count + 1
    return actor_par_count
