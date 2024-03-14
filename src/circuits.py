import math

import numpy as np
import pennylane as qml
import torch

from args import parse_args
from transform_funks import normalize_obs, transform_obs_to_binary

args = parse_args()


# tanh weight re-mapping
# according to   "Improving Convergence for Quantum Variational Classifiers using Weight Re-Mapping"  by Michael Kolle et al.


def tanh_remapping(layer_params):
    layer_params = np.pi * torch.tanh(layer_params)
    # intervall ]-pi,pi[
    return layer_params


#######################################################################################################################################################
##                                              quantum actor circuits                                                                               ##
##                                                                                                                                                   ##
dev = qml.device("default.qubit", wires=args.n_qubits)
# simple circuit:            (by Yunseok Kwak et all in Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation)


# Parameterized Rotation & Entanglement Layers
def simple_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RX(layer_params[i, layer_nr, 0], wires=i)
        qml.RY(layer_params[i, layer_nr, 1], wires=i)
        qml.RZ(layer_params[i, layer_nr, 2], wires=i)

    if (
        (layer_nr == args.n_var_layers - 1)
        and (args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1")
        and args.n_qubits == 4
    ):
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
    else:
        for i in range(args.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


# Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_actor_circuit(layer_params, input_scaleing_params, observation, act_dim):
    norm_obs = normalize_obs(observation)

    # Input Encoding
    for i in range(args.n_qubits):
        qml.RY(np.pi * norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [
            qml.expval(qml.PauliZ(ind)) for ind in range((args.n_qubits - act_dim), args.n_qubits)
        ]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_actor_circuit(layer_params, input_scaleing_params, observation, act_dim):
    norm_obs = normalize_obs(observation)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        # Encodeing layer
        for i in range(args.n_qubits):
            qml.RY(np.pi * norm_obs[i], wires=i)
        # Variational Layer
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [
            qml.expval(qml.PauliZ(ind)) for ind in range((args.n_qubits - act_dim), args.n_qubits)
        ]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_actor_circuit_with_shared_input_scaleing(
    layer_params, input_scaleing_params, observation, act_dim
):
    for layer_nr in range(args.n_var_layers):
        # Encodeing layer
        # for discrete states transform obs to binary
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i] * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi * torch.tanh(input_scaleing_params[i] * observation[i]),
                    wires=i,
                )
        # Variational Layer
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [
            qml.expval(qml.PauliZ(ind)) for ind in range((args.n_qubits - act_dim), args.n_qubits)
        ]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_actor_circuit_with_input_scaleing(
    layer_params, input_scaleing_params, observation, act_dim
):
    for layer_nr in range(args.n_var_layers):
        # Encodeing layer
        # for discrete states transform obs to binary
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i, layer_nr]
                        * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi * torch.tanh(input_scaleing_params[i, layer_nr] * observation[i]),
                    wires=i,
                )
        # Variational Layer
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [
            qml.expval(qml.PauliZ(ind)) for ind in range((args.n_qubits - act_dim), args.n_qubits)
        ]


# Hgog circuit:            (by Mohamad Hgog in: Quantum-Enhanced Policy Gradient Methods for Reinforcement Learning)


# Parameterized Rotation & Entanglement Layers
def Hgog_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RZ(layer_params[i, layer_nr, 0], wires=i)
        qml.RY(layer_params[i, layer_nr, 1], wires=i)
        qml.RZ(layer_params[i, layer_nr, 2], wires=i)

    for i in range(args.n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[args.n_qubits - 1, 0])


# Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_actor_circuit(layer_params, input_scaleing_params, observation, act_dim):
    norm_obs = normalize_obs(observation)
    # Input Encoding
    for i in range(args.n_qubits):
        qml.RX(np.pi * norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(act_dim)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_reuploading_actor_circuit(layer_params, input_scaleing_params, observation, act_dim):
    norm_obs = normalize_obs(observation)

    for layer_nr in range(args.n_var_layers):
        # Encodeing Layer
        for i in range(args.n_qubits):
            qml.RX(np.pi * norm_obs[i], wires=i)
        # Variational Layer
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(act_dim)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_reuploading_actor_circuit_with_input_scaleing(
    layer_params, input_scaleing_params, observation, act_dim
):
    for layer_nr in range(args.n_var_layers):
        # Encodeing layer
        # for discrete states transform obs to binary
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            # get 1D input scaleing parameters from 3D array
            for i in range(args.n_qubits):
                qml.RX(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i, layer_nr]
                        * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            # get 1D input scaleing parameters from 3D array
            for i in range(args.n_qubits):
                qml.RX(
                    np.pi * torch.tanh(input_scaleing_params[i, layer_nr] * observation[i]),
                    wires=i,
                )
        # Variational Layer
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(act_dim)]


# Jerbi circuit:             (by Jerbi et al. in: Parametrized Quantum Policies for Reinforcement Learning)


def variational_layer(layer_params, layer_nr):
    for i in range(args.n_qubits):
        qml.RZ(layer_params[i, layer_nr, 0], wires=i)
        qml.RY(layer_params[i, layer_nr, 1], wires=i)
    for i in range(args.n_qubits - 1):
        qml.CZ(wires=[i, i + 1])
    if args.n_qubits != 2:
        qml.CZ(wires=[args.n_qubits - 1, 0])

    # for i in range(args.n_qubits - 1):
    #    qml.CNOT(wires=[i, i + 1])
    # qml.CNOT(wires=[args.n_qubits - 1, 0])


def encodeing_layer(encodeing_params, layer_nr, state_vector):
    for i in range(args.n_qubits):
        qml.RY(encodeing_params[i, layer_nr, 0] * state_vector[i], wires=i)
        qml.RZ(encodeing_params[i, layer_nr, 1] * state_vector[i], wires=i)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_reuploading_actor_circuit(layer_params, input_scaleing_params, observation, act_dim):
    # wire preparation
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    for layer_nr in range(args.n_var_layers - 1):
        # for envs with discrete states transform obs to binary
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            encodeing_layer(
                tanh_remapping(input_scaleing_params),
                layer_nr,
                transform_obs_to_binary(observation),
            )
        else:
            encodeing_layer(tanh_remapping(input_scaleing_params), layer_nr, observation)
        variational_layer(tanh_remapping(layer_params), layer_nr + 1)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_actor_circuit_no_reuploading_no_input_scaleing(
    layer_params, input_scaleing_params, observation, act_dim
):
    norm_obs = normalize_obs(observation)

    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    for i in range(args.n_qubits):
        qml.RY(np.pi * norm_obs[i], wires=i)
        qml.RZ(np.pi * norm_obs[i], wires=i)

    for layer_nr in range(1, args.n_var_layers):
        variational_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_reuploading_actor_circuit_without_input_scaleing(
    layer_params, input_scaleing_params, observation, act_dim
):
    norm_obs = normalize_obs(observation)
    # layer_params: Variable Layer Parameters, observation: State Variable
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    for layer_nr in range(1, args.n_var_layers):
        # encodeing layer
        for i in range(args.n_qubits):
            qml.RY(np.pi * norm_obs[i], wires=i)
            qml.RZ(np.pi * norm_obs[i], wires=i)
        variational_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(ind)) for ind in range(act_dim)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def empty_circuit(layer_params, observation, act_dim=None):
    return


##                                                                                                                                                   ##
##                                              quantum actor circuits                                                                               ##
#######################################################################################################################################################


def actor_circuit_selection():
    # circuit selection
    if args.quantum_actor and args.circuit == "simple":
        actor_circuit = simple_actor_circuit
        print("useing simple Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "simple_reuploading":
        actor_circuit = simple_reuploading_actor_circuit
        print("useing simple reuploading Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "simple_reuploading_with_shared_input_scaleing":
        actor_circuit = simple_reuploading_actor_circuit_with_shared_input_scaleing
        print("useing simple reuploading Quantum Circuit with input scaleing as actor")
    elif args.quantum_actor and args.circuit == "simple_reuploading_with_input_scaleing":
        actor_circuit = simple_reuploading_actor_circuit_with_input_scaleing
        print("useing simple reuploading Quantum Circuit with input scaleing as actor")

    elif args.quantum_actor and args.circuit == "Hgog":
        actor_circuit = Hgog_actor_circuit
        print("useing Hgog Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "Hgog_reuploading":
        actor_circuit = Hgog_reuploading_actor_circuit
        print("useing Hgog reuploading Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "Hgog_reuploading_with_input_scaleing":
        actor_circuit = Hgog_reuploading_actor_circuit_with_input_scaleing
        print("useing Hgog reuploading Quantum Circuit with input scaleing as actor")

    elif args.quantum_actor and args.circuit == "Jerbi-no-reuploading-no-input-scaleing":
        actor_circuit = Jerbi_actor_circuit_no_reuploading_no_input_scaleing
        print("useing Jerbi simple Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "Jerbi-reuploading":
        actor_circuit = Jerbi_reuploading_actor_circuit
        print("useing Jerbi reuploading Quantum Circuit as actor")
    elif args.quantum_actor and args.circuit == "Jerbi-reuploading-no-input-scaleing":
        actor_circuit = Jerbi_reuploading_actor_circuit_without_input_scaleing
        print("useing Jerbi reuploading Quantum Circuit without input scaleing as actor")
    elif not args.quantum_actor:
        actor_circuit = empty_circuit
        print("useing classical actor")
    else:
        print("actor selection ERROR: ", args.circuit, " does not exist")

    if args.quantum_actor and args.hybrid:
        print("with hybrid input postprocessing")
    elif args.quantum_actor and args.output_scaleing:
        print("with output scaleing")
    elif args.quantum_actor:
        print("without input scaleing")

    return actor_circuit


#######################################################################################################################################################
##                                              quantum critic circuits                                                                              ##
##
#                                                                                                                                                  ##
# simple circuit:            (by Yunseok Kwak et all in Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation)


# Variational Quantum Policy Circuit (Critic)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_critic_circuit(layer_params, input_scaleing_params, observation):
    norm_obs = normalize_obs(observation)

    # Input Encoding
    for i in range(args.n_qubits):
        qml.RY(np.pi * norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_critic_circuit(layer_params, input_scaleing_params, observation):
    norm_obs = normalize_obs(observation)

    for layer_nr in range(args.n_var_layers):
        for i in range(args.n_qubits):
            qml.RY(np.pi * norm_obs[i], wires=i)
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_critic_circuit_with_shared_input_scaleing(
    layer_params, input_scaleing_params, observation
):
    for layer_nr in range(args.n_var_layers):
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i] * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            for i in range(args.n_qubits):
                qml.RY(np.pi * torch.tanh(input_scaleing_params[i] * observation[i]), wires=i)
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def simple_reuploading_critic_circuit_with_input_scaleing(
    layer_params, input_scaleing_params, observation
):
    for layer_nr in range(args.n_var_layers):
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i, layer_nr]
                        * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            for i in range(args.n_qubits):
                qml.RY(
                    np.pi * torch.tanh(input_scaleing_params[i, layer_nr] * observation[i]),
                    wires=i,
                )
        simple_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


# Hgog simple circuit:            (by Mohamad Hgog in: Quantum-Enhanced Policy Gradient Methods for Reinforcement Learning)


# Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_critic_circuit(layer_params, input_scaleing_params, observation):
    norm_obs = normalize_obs(observation)
    # Input Encoding
    for i in range(args.n_qubits):
        qml.RX(np.pi * norm_obs[i], wires=i)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_reuploading_critic_circuit(layer_params, input_scaleing_params, observation):
    norm_obs = normalize_obs(observation)

    # Variational Quantum Circuit
    for layer_nr in range(args.n_var_layers):
        # encodeing layer
        for i in range(args.n_qubits):
            qml.RX(np.pi * norm_obs[i], wires=i)
        # variational layer
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Hgog_reuploading_critic_circuit_with_input_scaleing(
    layer_params, input_scaleing_params, observation
):
    for layer_nr in range(args.n_var_layers):
        if (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):
            for i in range(args.n_qubits):
                qml.RX(
                    np.pi
                    * torch.tanh(
                        input_scaleing_params[i, layer_nr]
                        * transform_obs_to_binary(observation)[i]
                    ),
                    wires=i,
                )
        else:
            for i in range(args.n_qubits):
                qml.RX(
                    np.pi * torch.tanh(input_scaleing_params[i, layer_nr] * observation[i]),
                    wires=i,
                )
        Hgog_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliZ(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliZ(args.n_qubits - 1))]


# Jerbi circuit:


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_reuploading_critic_circuit(layer_params, input_scaleing_params, observation):
    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
    ):
        for layer_nr in range(args.n_var_layers - 1):
            encodeing_layer(
                tanh_remapping(input_scaleing_params),
                layer_nr,
                transform_obs_to_binary(observation),
            )
            variational_layer(tanh_remapping(layer_params), layer_nr + 1)
    else:
        for layer_nr in range(args.n_var_layers - 1):
            encodeing_layer(tanh_remapping(input_scaleing_params), layer_nr, observation)
            variational_layer(tanh_remapping(layer_params), layer_nr + 1)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]


# Variational Quantum Policy Circuit (Actor)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_critic_circuit_no_reuploading_no_input_scaleing(
    layer_params, input_scaleing_params, observation
):
    norm_obs = normalize_obs(observation)

    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    for i in range(args.n_qubits):
        qml.RY(np.pi * norm_obs[i], wires=i)
        qml.RZ(np.pi * norm_obs[i], wires=i)

    for layer_nr in range(1, args.n_var_layers):
        variational_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def Jerbi_reuploading_critic_circuit_without_input_scaleing(
    layer_params, input_scaleing_params, observation
):
    norm_obs = normalize_obs(observation)

    for i in range(args.n_qubits):
        qml.Hadamard(wires=i)

    variational_layer(tanh_remapping(layer_params), 0)

    for layer_nr in range(1, args.n_var_layers):
        for i in range(args.n_qubits):
            qml.RY(np.pi * norm_obs[i], wires=i)
            qml.RZ(np.pi * norm_obs[i], wires=i)
        variational_layer(tanh_remapping(layer_params), layer_nr)

    if args.hybrid:
        return [qml.expval(qml.PauliY(ind)) for ind in range(args.n_qubits)]
    else:
        return [qml.expval(qml.PauliY(args.n_qubits - 1))]


##                                                                                                                                                   ##
##                                              quantum critic circuits                                                                              ##
#######################################################################################################################################################


def critic_circuit_selection():
    if args.quantum_critic and args.circuit == "simple":
        critic_circuit = simple_critic_circuit
        print("useing simple Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "simple_reuploading":
        critic_circuit = simple_reuploading_critic_circuit
        print("useing simple reuploading Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "simple_reuploading_with_shared_input_scaleing":
        critic_circuit = simple_reuploading_critic_circuit_with_shared_input_scaleing
        print("useing simple reuploading Quantum Circuit with shared input scaleing as critic")
    elif args.quantum_critic and args.circuit == "simple_reuploading_with_input_scaleing":
        critic_circuit = simple_reuploading_critic_circuit_with_input_scaleing
        print("useing simple reuploading Quantum Circuit with input scaleing as critic")

    elif args.quantum_critic and args.circuit == "Hgog":
        critic_circuit = Hgog_critic_circuit
        print("useing Hagog Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "Hgog_reuploading":
        critic_circuit = Hgog_reuploading_critic_circuit
        print("useing Hgog reuploading Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "Hgog_reuploading_with_input_scaleing":
        critic_circuit = Hgog_reuploading_critic_circuit_with_input_scaleing
        print("useing Hgog reuploading Quantum Circuit with input scaleing as critic")

    elif args.quantum_critic and args.circuit == "Jerbi-no-reuploading-no-input-scaleing":
        critic_circuit = Jerbi_critic_circuit_no_reuploading_no_input_scaleing
        print("useing Jerbi simple Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "Jerbi-reuploading":
        critic_circuit = Jerbi_reuploading_critic_circuit
        print("useing Jerbi data reuploading Quantum Circuit as critic")
    elif args.quantum_critic and args.circuit == "Jerbi-reuploading-no-input-scaleing":
        critic_circuit = Jerbi_reuploading_critic_circuit_without_input_scaleing
        print("useing Jerbi reuploading Quantum Circuit without input scaleing as critic")
    elif not args.quantum_critic:
        critic_circuit = empty_circuit
        print("useing classical critic")
    else:
        print("critic selection ERROR: ", args.circuit, " does not exist")

    if args.quantum_critic and args.hybrid:
        print("with hybrid input postprocessing")
    elif args.quantum_critic and not args.hybrid:
        print("with manual input rescaleing")

    return critic_circuit
