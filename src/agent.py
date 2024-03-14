import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from args import parse_args
from transform_funks import trans_obs
from utils import get_act_dim, get_obs_dim

args = parse_args()
#######################################################################################################################################################
##                                                  agent:                                                                                           ##
##                                                                                                                                                   ##


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # 2 Layer Initialisation
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        if not args.quantum_critic:
            self.critic = nn.Sequential(
                layer_init(
                    nn.Linear(
                        np.array(get_obs_dim(envs.single_observation_space)).prod(),
                        args.critic_hidden_layer_nodes,
                    )
                ),
                nn.Tanh(),
                layer_init(
                    nn.Linear(args.critic_hidden_layer_nodes, args.critic_hidden_layer_nodes)
                ),
                nn.Tanh(),
                layer_init(nn.Linear(args.critic_hidden_layer_nodes, 1), std=1.0),
            )

        if not args.quantum_actor and args.actor_hidden_layer2_nodes != 0:
            self.actor = nn.Sequential(
                layer_init(
                    nn.Linear(
                        np.array(get_obs_dim(envs.single_observation_space)).prod(),
                        args.actor_hidden_layer1_nodes,
                    )
                ),
                nn.Tanh(),
                layer_init(
                    nn.Linear(args.actor_hidden_layer1_nodes, args.actor_hidden_layer2_nodes)
                ),
                nn.Tanh(),
                layer_init(
                    nn.Linear(
                        args.actor_hidden_layer2_nodes, get_act_dim(envs.single_action_space)
                    ),
                    std=0.01,
                ),
            )
        elif not args.quantum_actor:
            self.actor = nn.Sequential(
                layer_init(
                    nn.Linear(
                        np.array(get_obs_dim(envs.single_observation_space)).prod(),
                        args.actor_hidden_layer1_nodes,
                    )
                ),
                nn.Tanh(),
                layer_init(
                    nn.Linear(
                        args.actor_hidden_layer1_nodes, get_act_dim(envs.single_action_space)
                    ),
                    std=0.01,
                ),
            )

        if args.hybrid:
            self.hybrid_actor = nn.Sequential(
                layer_init(nn.Linear(args.n_qubits, get_act_dim(envs.single_action_space)))
            )
            self.hybrid_critic = nn.Sequential(layer_init(nn.Linear(args.n_qubits, 1)))

        self.checkpoint_file = os.path.join(args.chkpt_dir, "classical_network_params")

    def save_classical_network_params(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_classical_network_params(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def get_value(
        self,
        observation,
        obs_dim,
        critic_circuit,
        critic_layer_params,
        critic_input_scaleing_params,
    ):
        if args.quantum_critic:  # quantum critic
            crit_tmp = critic_circuit(
                critic_layer_params, critic_input_scaleing_params, observation
            )

            if args.hybrid:  # hybrid output rescaleing
                crit = self.hybrid_critic(crit_tmp.float())
            else:  # manual output rescaleing
                if args.gym_id == "FrozenLake-v0" or args.gym_id == "FrozenLake-v1":
                    crit = (crit_tmp + 1) / 2  # rescaleing from [-1,1] to [0,1]
                elif (
                    args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
                    or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
                ):
                    crit = crit_tmp  # intervall [-1,1]
                elif args.gym_id == "CartPole-v0":
                    crit = (crit_tmp + 1) * 100  # rescaleing from [-1,1] to [0,200]
                elif args.gym_id == "CartPole-v1":
                    crit = (crit_tmp + 1) * 250  # rescaleing from [-1,1] to [0,500]
                else:
                    print("manual output rescaleing ERROR for value Function")
                    crit = crit_tmp
                    raise NotImplementedError()
        else:  # classical critic
            crit = self.critic(trans_obs(observation, obs_dim))

        return crit

    def get_action_and_value(
        self,
        observation,
        actor_circuit,
        actor_layer_params,
        actor_input_scaleing_params,
        critic_circuit,
        critic_layer_params,
        critic_input_scaleing_params,
        output_scaleing_params,
        obs_dim,
        acts_dim,
        action=None,
    ):
        if args.quantum_actor:
            logits_uncat = actor_circuit(
                actor_layer_params, actor_input_scaleing_params, observation, acts_dim
            )  # logits_uncat is a list, logits needs to be a tensor

            if args.hybrid:  # hybrid output postprocessing
                logits_cat = torch.Tensor(acts_dim)
                for i in range(acts_dim):
                    logits_cat[i] = logits_uncat[i]
                logits = self.hybrid_actor(logits_cat.float())

            elif args.output_scaleing:
                # output scaleing with (trainable) output parameters for (trainable) greedyness
                logits = torch.zeros(acts_dim)
                for i in range(acts_dim):
                    if (
                        not args.shared_output_scaleing_param
                        # and not args.scheduled_output_scaleing
                    ):
                        logits[i] = logits_uncat[i] * (output_scaleing_params[i])
                    else:
                        logits[i] = logits_uncat[i] * (output_scaleing_params[0])
                    # * output_scaleing_params.mean()
            else:
                # merge list of tensors to tensor
                logits = torch.Tensor(acts_dim)
                for i in range(acts_dim):
                    logits[i] = logits_uncat[i]
        else:
            logits = self.actor(trans_obs(observation, obs_dim))

        if args.quantum_actor and not args.output_scaleing and not args.hybrid:
            probs = Categorical(logits + 1)  # softMaxOutPut = (logits+1) / (logits+1).sum()

        else:
            probs = Categorical(logits=logits)
            # softMaxOutPut = exp(logits) / exp(logits).sum()
        if action is None:
            action = probs.sample()
            action = action.view(1)

        logprobs = probs.log_prob(action)
        entropy = probs.entropy()

        return (
            action,
            logprobs,
            entropy,
            self.get_value(
                observation,
                obs_dim,
                critic_circuit,
                critic_layer_params,
                critic_input_scaleing_params,
            ),
        )

    def get_random_action_and_value(self, acts_dim, action=None):
        logits = torch.ones(acts_dim)
        probs = Categorical(logits=logits)  # softMaxOutPut = np.exp(logits) / np.exp(logits).sum()
        if action is None:
            action = probs.sample()
            action = action.view(1)

        value = torch.zeros(1)

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            value,
        )


##                                                                                                                                                   ##
##                                                  agent:                                                                                           ##
#######################################################################################################################################################
