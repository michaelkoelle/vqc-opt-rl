import numpy as np
import torch

from args import parse_args

args = parse_args()


def transform_obs_to_one_hot_encodeing(observation, single_observation_space_n):
    one_hot_encodeing_observation = torch.zeros(single_observation_space_n)
    one_hot_encodeing_observation[int(observation)] = 1.0
    return one_hot_encodeing_observation


def transform_obs_to_binary(observation):
    binary_observation = torch.zeros(args.n_qubits)
    binary = str(bin(int(observation)))
    binary = binary[2:]
    length = len(binary)

    for i in range(args.n_qubits):
        if (length) <= (args.n_qubits - (i + 1)):
            binary_observation[i] = 0.0
        else:
            binary_observation[i] = float(binary[length - 1 - (args.n_qubits - (i + 1))])
    return binary_observation


def trans_obs(observation, single_observation_space_n):
    if (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
    ):
        trans_obs_ = transform_obs_to_one_hot_encodeing(observation, single_observation_space_n)
    else:
        trans_obs_ = observation
    return trans_obs_


def normalize_obs(observation):
    if (
        args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1"
    ) and args.n_qubits == 4:  # specific normalisation for cartpole      terminates if |position|>2.4 or |angle|>0.2095
        norm_obs = observation
        if args.arctan_input_rescale:
            norm_obs = torch.Tensor(
                [
                    torch.clamp((norm_obs[0] / 4.8), -1.0, 1.0),
                    2 * torch.arctan(norm_obs[1]) / np.pi,
                    torch.clamp((norm_obs[2] / 0.418), -1.0, 1.0),
                    2 * torch.arctan(norm_obs[3]) / np.pi,
                ]
            )
        else:
            norm_obs = torch.Tensor(
                [
                    torch.clamp((norm_obs[0] / 4.8), -1.0, 1.0),
                    torch.tanh(norm_obs[1]),
                    torch.clamp((norm_obs[2] / 0.418), -1.0, 1.0),
                    torch.tanh(norm_obs[3]),
                ]
            )
    elif (
        args.gym_id == "FrozenLake-v0"
        or args.gym_id == "FrozenLake-v1"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
        or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
    ):
        norm_obs = transform_obs_to_binary(observation)
    else:
        norm_obs = torch.tanh(observation)
        # print("normalisation Error: no handcrafted normalisation")
    return norm_obs
