# ppo based on https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import save_params
from agent import Agent
from args import parse_args, save_args
from calc_num_params import calc_num_actor_params, calc_num_critic_params
from circuits import actor_circuit_selection, critic_circuit_selection
from env_setup import make_env
from envs_storage import Store_envs
from layer_params import make_actor_layer_params, make_critic_layer_params
from plot_old import plot_training_results
from save_results import Save_results
from utils import get_act_dim, get_obs_dim

args = parse_args()
chkpt_pathExists = os.path.exists(args.chkpt_dir)
if not chkpt_pathExists:
    os.makedirs(args.chkpt_dir)
results_pathExists = os.path.exists(args.results_dir)
if not results_pathExists:
    os.makedirs(args.results_dir)

save_args(args)

if __name__ == "__main__":
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Setup env
    store_envs = Store_envs()
    if args.load_chkpt:
        store_envs.load_envs(args.chkpt_dir, args.num_envs)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.gym_id,
                args.seed + i,
                i,
                args.num_envs,
                args.chkpt_dir,
                args.load_chkpt,
                store_envs,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = gym.wrappers.VectorListInfo(envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "onely discrete action spaces supported"
    save_results = Save_results(
        args.results_dir, args.load_chkpt, args.record_grads
    )

    actor_par_count = calc_num_actor_params(envs)
    critic_par_count = calc_num_critic_params(envs)
    print("calculated number of actor parameters: ", actor_par_count)
    print("calculated number of critic parameters: ", critic_par_count)

    # Declare Quantum Circuit Parameters and agent
    agent = Agent(envs)
    (
        actor_layer_params,
        actor_input_scaleing_params,
    ) = (
        make_actor_layer_params()
    )  # alternative: layer_params = np.random.normal(0, np.tan(0.5), (args.n_qubits, (2 * args.n_var_layers) + 2, 2))
    if not args.shared_output_scaleing_param:  # and not args.scheduled_output_scaleing
        output_scaleing_params = (
            np.ones(get_act_dim(envs.single_action_space)) * args.output_scaleing_start
        )
    else:
        output_scaleing_params = np.ones(1) * args.output_scaleing_start
    critic_layer_params, critic_input_scaleing_params = make_critic_layer_params()

    if args.load_chkpt:  # load Parameters from checkpoint
        if not args.quantum_actor or not args.quantum_critic:
            agent.load_classical_network_params()
        if args.quantum_actor:
            actor_layer_params = save_params.load_actor_circuit_params(args.chkpt_dir)
            actor_input_scaleing_params = save_params.load_actor_input_scaleing_params(
                args.chkpt_dir
            )
        if (
            args.quantum_actor
            and not args.hybrid
            and args.output_scaleing
            # and not args.scheduled_output_scaleing
        ):
            output_scaleing_params = save_params.load_output_scaleing_params(args.chkpt_dir)
        else:
            output_scaleing_params = Variable(
                torch.tensor(output_scaleing_params), requires_grad=False
            )
        if args.quantum_critic:
            critic_layer_params = save_params.load_critic_circuit_params(args.chkpt_dir)
            actor_input_scaleing_params = save_params.load_critic_input_scaleing_params(
                args.chkpt_dir
            )

    else:  # make existing Parameters trainable
        actor_layer_params = Variable(torch.tensor(actor_layer_params), requires_grad=True)
        actor_input_scaleing_params = Variable(
            torch.tensor(actor_input_scaleing_params), requires_grad=True
        )
        if (
            args.quantum_actor
            and not args.hybrid
            and args.output_scaleing
            # and not args.scheduled_output_scaleing
        ):
            output_scaleing_params = Variable(
                torch.tensor(output_scaleing_params), requires_grad=True
            )
        else:
            output_scaleing_params = Variable(
                torch.tensor(output_scaleing_params), requires_grad=False
            )
        critic_layer_params = Variable(torch.tensor(critic_layer_params), requires_grad=True)
        critic_input_scaleing_params = Variable(
            torch.tensor(critic_input_scaleing_params), requires_grad=True
        )

    actor_circuit = actor_circuit_selection()
    critic_circuit = critic_circuit_selection()

    # classic_agent_optimizers init
    if not args.quantum_actor:
        actor_optimizer = optim.Adam(
            agent.actor.parameters(), lr=args.classic_actor_learning_rate, eps=1e-5
        )
    elif args.hybrid:
        hybrid_actor_optimizer = optim.Adam(
            agent.hybrid_actor.parameters(), lr=args.classic_actor_learning_rate, eps=1e-5
        )

    if not args.quantum_critic:
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.hybrid:
        hybrid_critic_optimizer = optim.Adam(
            agent.hybrid_critic.parameters(), lr=args.learning_rate, eps=1e-5
        )

    # circuit_optimizers init
    if args.quantum_actor:  # 3 Different Adams epsylon
        quantum_actor_optimizer = optim.Adam(
            [actor_layer_params], lr=args.qlearning_rate, eps=1e-5
        )
    if (
        args.quantum_actor
        and not args.hybrid
        and args.output_scaleing
        # and not args.scheduled_output_scaleing
    ):
        output_scaleing_optimizer = optim.Adam(
            [output_scaleing_params], lr=args.output_scaleing_learning_rate, eps=1e-5
        )

    if args.quantum_critic:
        quantum_critic_optimizer = optim.Adam(
            [critic_layer_params], lr=args.qlearning_rate, eps=1e-5
        )

    if args.circuit == "simple_reuploading_with_shared_input_scaleing":
        # input_scaleing_learning_rate=output_scaleing_learning_rate
        if args.quantum_actor:
            actor_input_scaleing_optimizer = optim.Adam(
                [actor_input_scaleing_params], lr=args.output_scaleing_learning_rate, eps=1e-5
            )
        if args.quantum_critic:
            critic_input_scaleing_optimizer = optim.Adam(
                [critic_input_scaleing_params], lr=args.output_scaleing_learning_rate, eps=1e-5
            )

    if (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
        or args.circuit == "Jerbi-reuploading"
    ):
        # input_scaleing_learning_rate=qlearning_rate (inclusive scheduling)
        if args.quantum_actor:
            actor_input_scaleing_optimizer = optim.Adam(
                [actor_input_scaleing_params], lr=args.qlearning_rate, eps=1e-5
            )
        if args.quantum_critic:
            critic_input_scaleing_optimizer = optim.Adam(
                [critic_input_scaleing_params], lr=args.qlearning_rate, eps=1e-5
            )

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.short
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    # Game start
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    exp_scheduling_updates = args.exp_scheduling_halftime // args.batch_size
    # sq_scheduling_updates = args.sq_scheduling_timesteps // args.batch_size
    # lin_scheduling_updates = args.lin_scheduling_timesteps // args.batch_size

    if args.load_chkpt:
        global_step, next_obs, next_done = save_params.load_state(args.chkpt_dir)
        if not (
            args.gym_id == "FrozenLake-v0"
            or args.gym_id == "FrozenLake-v1"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0"
            or args.gym_id == "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
        ):  # returning to the old env state after restart onely works in Frozen Lake
            obs_tmp, _ = envs.reset()
            next_obs = torch.Tensor(obs_tmp)
            next_done = torch.zeros(args.num_envs)

        # updates_remaining = (args.total_timesteps - global_step) // args.batch_size
        done_updates = (global_step) // args.batch_size
    else:
        global_step = 0
        obs_tmp, _ = envs.reset()
        next_obs = torch.Tensor(obs_tmp)
        next_done = torch.zeros(args.num_envs)
        done_updates = 0

    # Training loop
    for update in range(done_updates + 1, num_updates + 1):
        # Annealing the Quantum Leaning Rate if activated

        if args.exp_qlr_scheduling:
            frac_exp = 2.0 ** ((-update + 1.0) / exp_scheduling_updates)
            if args.quantum_actor:
                qlrnow_circuit = (
                    frac_exp * (args.exp_scheduling_qlearning_rate - args.qlearning_rate)
                    + args.qlearning_rate
                )
            else:
                lrnow_circuit = (
                    frac_exp * (args.exp_scheduling_qlearning_rate - args.classic_actor_learning_rate)
                    + args.classic_actor_learning_rate
                )
        elif args.sigmoid_lr_scheduling:     #sigmoid lr anneahling after Nakamura et al.
            t=10*((update-1.0)/exp_scheduling_updates)
            frac_sigmoid = 1/(1+np.exp(0.5*(2*t-10)))
            if args.quantum_actor:
                qlrnow_circuit = (
                    frac_sigmoid * (args.exp_scheduling_qlearning_rate - args.qlearning_rate)
                    + args.qlearning_rate
                )
            else:
                lrnow_circuit = (
                    frac_sigmoid * (args.exp_scheduling_qlearning_rate - args.classic_actor_learning_rate)
                    + args.classic_actor_learning_rate
                )


        if args.exp_qlr_scheduling or args.sigmoid_lr_scheduling:
            if args.quantum_actor:
                quantum_actor_optimizer.param_groups[0]["lr"] = qlrnow_circuit
            else: 
                actor_optimizer.param_groups[0]["lr"] = lrnow_circuit
            #if args.quantum_critic:
            #    quantum_critic_optimizer.param_groups[0]["lr"] = qlrnow_circuit
            if (
                args.circuit == "simple_reuploading_with_input_scaleing"
                or args.circuit == "Hgog_reuploading_with_input_scaleing"
                or args.circuit == "Jerbi-reuploading"
            ):
                if args.quantum_actor:
                    actor_input_scaleing_optimizer.param_groups[0]["lr"] = qlrnow_circuit

                #if args.quantum_critic:
                #    critic_input_scaleing_optimizer.param_groups[0]["lr"] = qlrnow_circuit

        # Environment interaction
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action
            for i in range(args.num_envs):
                with torch.no_grad():
                    if args.random_baseline:
                        action, logprob, _, value = agent.get_random_action_and_value(
                            get_act_dim(envs.single_action_space)
                        )
                    else:
                        action, logprob, _, value = agent.get_action_and_value(
                            next_obs[i],
                            actor_circuit,
                            actor_layer_params,
                            actor_input_scaleing_params,
                            critic_circuit,
                            critic_layer_params,
                            critic_input_scaleing_params,
                            output_scaleing_params,
                            get_obs_dim(envs.single_observation_space),
                            get_act_dim(envs.single_action_space),
                        )
                    values[step, i] = value.flatten()
                actions[step, i] = action.short()
                logprobs[step, i] = logprob

            # Env step
            next_ob, reward, terminated, truncated, info = envs.step(
                actions[step].cpu().numpy()
            )  # np.asarray(x, dtype = 'int')   #

            rewards[step] = torch.tensor(reward).view(-1)
            done = torch.logical_or(torch.Tensor(terminated), torch.Tensor(truncated))
            next_obs, next_done = torch.Tensor(next_ob), done
            for item in info:
                if "final_info" in item.keys():
                    print(
                        f"global_step={global_step}, episodic return {item['final_info']['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", item["final_info"]["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", item["final_info"]["episode"]["l"], global_step
                    )
                    save_results.append_episode_results(
                        item["final_info"]["episode"]["r"],
                        item["final_info"]["episode"]["l"],
                        global_step,
                        args.gym_id,
                        args.exp_name,
                        args.circuit,
                        args.seed,
                    )

        if not args.random_baseline:
            # calculate commulative discounted Returns and Advantages
            with torch.no_grad():
                next_value_i = torch.zeros(args.num_envs)
                for i in range(args.num_envs):
                    next_value_i[i] = agent.get_value(
                        next_obs[i],
                        get_obs_dim(envs.single_observation_space),
                        critic_circuit,
                        critic_layer_params,
                        critic_input_scaleing_params,
                    )
                next_value = next_value_i.reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards)
                    nextadvantage = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done.short()
                            nextvalue = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1].short()
                            nextvalue = values[t + 1]
                        delta = (
                            rewards[t] + args.gamma * nextnonterminal * nextvalue - values[t]
                        )  # analog zu returns[t] - values[t] nur mit nextvalue anstatt next_return
                        advantages[t] = nextadvantage = (
                            delta + args.gamma * args.gae_lambda * nextnonterminal * nextadvantage
                        )  # + commulative discounted advantage
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done.short()
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1].short()
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + args.gamma * nextnonterminal * next_return
                        )  # commulative discounted return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Start optimizing the policy and value network
            b_inds = np.arange(args.batch_size)  # Make batches
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)  # Shuffle batches
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]  # Make minibatches
                    # Start training,
                    newlogprob = torch.zeros(args.minibatch_size)
                    entropy = torch.zeros(args.minibatch_size)
                    newvalue = torch.zeros(args.minibatch_size)

                    for i in range(args.minibatch_size):  # Make minibatches        #problem 4 size
                        # Get values, ratio
                        _, newlogprob_i, entropy_i, newvalue_i = agent.get_action_and_value(
                            b_obs[mb_inds[i]],
                            actor_circuit,
                            actor_layer_params,
                            actor_input_scaleing_params,
                            critic_circuit,
                            critic_layer_params,
                            critic_input_scaleing_params,
                            output_scaleing_params,
                            get_obs_dim(envs.single_observation_space),
                            get_act_dim(envs.single_action_space),
                            b_actions.long()[mb_inds[i]],
                        )

                        newlogprob[i] = newlogprob_i
                        entropy[i] = entropy_i
                        newvalue[i] = newvalue_i

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Advantage normalisation
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # debug variables (info)
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = (
                            (ratio - 1) - logratio
                        ).mean()  # the approximate Kullback–Leibler divergence: how agressice does the policy update
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]  # the fraction of the training data that triggered the clipped objective

                    # Start loss calcualtion
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    # Overall loss
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    # classic_agent_optimizers.zero_grad()
                    if not args.quantum_actor:
                        actor_optimizer.zero_grad()
                    elif args.hybrid:
                        hybrid_actor_optimizer.zero_grad()

                    if not args.quantum_critic:
                        critic_optimizer.zero_grad()
                    elif args.hybrid:
                        hybrid_critic_optimizer.zero_grad()

                    # circuit optimizers.zero_grad()
                    if args.quantum_actor:
                        quantum_actor_optimizer.zero_grad()
                        if (
                            not args.hybrid
                            and args.output_scaleing
                            # and not args.scheduled_output_scaleing
                        ):
                            output_scaleing_optimizer.zero_grad()
                    if args.quantum_critic:
                        quantum_critic_optimizer.zero_grad()
                    if (
                        args.circuit == "simple_reuploading_with_shared_input_scaleing"
                        or args.circuit == "simple_reuploading_with_input_scaleing"
                        or args.circuit == "Hgog_reuploading_with_input_scaleing"
                        or args.circuit == "Jerbi-reuploading"
                    ):
                        if args.quantum_actor:
                            actor_input_scaleing_optimizer.zero_grad()
                        if args.quantum_critic:
                            critic_input_scaleing_optimizer.zero_grad()

                    loss.backward()

                    # logging of gradients var and mean for plotting
                    if args.record_grads:
                        if args.quantum_actor:
                            actor_grads_abs_mean = torch.mean(torch.abs(actor_layer_params.grad))
                            actor_gradients_var = torch.var(actor_layer_params.grad)
                            actor_gradients_std = torch.std(actor_layer_params.grad)
                        else:
                            actor_grads_abs_mean = torch.tensor([0])
                            actor_gradients_var = torch.tensor([0])
                            actor_gradients_std = torch.tensor([0])

                        if args.quantum_critic:
                            critic_grads_abs_mean = torch.mean(torch.abs(critic_layer_params.grad))
                            critic_gradients_var = torch.var(critic_layer_params.grad)
                            critic_gradients_std = torch.std(critic_layer_params.grad)
                        else:
                            critic_grads_abs_mean = torch.tensor([0])
                            critic_gradients_var = torch.tensor([0])
                            critic_gradients_std = torch.tensor([0])
                        save_results.append_gradient_results(
                            actor_grads_abs_mean.item(),
                            actor_gradients_var.item(),
                            actor_gradients_std.item(),
                            critic_grads_abs_mean.item(),
                            critic_gradients_var.item(),
                            critic_gradients_std.item(),
                            global_step,
                            args.gym_id,
                            args.exp_name,
                            args.circuit,
                            args.seed,
                        )

                    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    if not args.quantum_actor:
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                    elif args.hybrid:
                        nn.utils.clip_grad_norm_(
                            agent.hybrid_actor.parameters(), args.max_grad_norm
                        )

                    if not args.quantum_critic:
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    elif args.hybrid:
                        nn.utils.clip_grad_norm_(
                            agent.hybrid_critic.parameters(), args.max_grad_norm
                        )

                    if args.quantum_actor and args.clip_circuit_grad_norm:
                        nn.utils.clip_grad_norm_([actor_layer_params], args.max_grad_norm)
                    if args.quantum_critic and args.clip_circuit_grad_norm:
                        nn.utils.clip_grad_norm_([critic_layer_params], args.max_grad_norm)
                    if (
                        args.circuit == "simple_reuploading_with_input_scaleing"
                        or args.circuit == "Hgog_reuploading_with_input_scaleing"
                        or args.circuit == "Jerbi-reuploading"
                    ) and args.clip_circuit_grad_norm:
                        if args.quantum_actor:
                            nn.utils.clip_grad_norm_(
                                [actor_input_scaleing_params], args.max_grad_norm
                            )
                        if args.quantum_critic:
                            nn.utils.clip_grad_norm_(
                                [critic_input_scaleing_params], args.max_grad_norm
                            )

                    # classic_agent_optimizers.step()
                    if not args.quantum_actor:
                        actor_optimizer.step()
                    elif args.hybrid:
                        hybrid_actor_optimizer.step()

                    if not args.quantum_critic:
                        critic_optimizer.step()
                    elif args.hybrid:
                        hybrid_critic_optimizer.step()

                    # circuit_optimizers.step()
                    if args.quantum_actor:
                        quantum_actor_optimizer.step()
                        if (
                            not args.hybrid
                            and args.output_scaleing
                            # and not args.scheduled_output_scaleing
                        ):
                            output_scaleing_optimizer.step()
                    if args.quantum_critic:
                        quantum_critic_optimizer.step()
                    if (
                        args.circuit == "simple_reuploading_with_shared_input_scaleing"
                        or args.circuit == "simple_reuploading_with_input_scaleing"
                        or args.circuit == "Hgog_reuploading_with_input_scaleing"
                        or args.circuit == "Jerbi-reuploading"
                    ):
                        if args.quantum_actor:
                            actor_input_scaleing_optimizer.step()
                        if args.quantum_critic:
                            critic_input_scaleing_optimizer.step()

            # debug variables (info)
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )  # is the value function a good indicator for the returns

            # record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", args.classic_actor_learning_rate, global_step
            )
            qlearning_rate = 0
            learning_rate = 0
            if args.quantum_actor:
                writer.add_scalar(
                    "charts/qlearning_rate",
                    quantum_actor_optimizer.param_groups[0]["lr"],
                    global_step,
                )
                qlearning_rate = quantum_actor_optimizer.param_groups[0]["lr"]
            else:
                learning_rate = actor_optimizer.param_groups[0]["lr"]
                if args.quantum_critic:
                    writer.add_scalar(
                        "charts/qlearning_rate",
                        quantum_critic_optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                    qlearning_rate = quantum_critic_optimizer.param_groups[0]["lr"]

            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/loss", loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            output_scaleing = 1
            if args.quantum_actor and not args.hybrid and args.output_scaleing:
                writer.add_scalar(
                    "greedyness/output_scaleing",
                    output_scaleing_params.mean().item(),
                    global_step,  # ** 2
                )
                with torch.no_grad():
                    output_scaleing = output_scaleing_params.mean().item()  # ** 2

            # store data of update
            save_results.append_update_results(
                learning_rate,
                qlearning_rate,
                v_loss.item(),
                pg_loss.item(),
                entropy_loss.item(),
                loss.item(),
                old_approx_kl.item(),
                approx_kl.item(),
                np.mean(clipfracs),
                explained_var,
                int(global_step / (time.time() - start_time)),
                output_scaleing,
                global_step,
                args.gym_id,
                args.exp_name,
                args.circuit,
                args.seed,
            )
            store_envs.store_envs(actions, dones, args.num_steps, args.num_envs)

            # if save intervall save everything to file
            if update % args.save_intervall == 0:
                save_results.save_results()
                if not args.quantum_actor or not args.quantum_critic:
                    agent.save_classical_network_params()
                if args.quantum_actor:
                    save_params.save_actor_circuit_params(args.chkpt_dir, actor_layer_params)
                    save_params.save_actor_input_scaleing_params(
                        args.chkpt_dir, actor_input_scaleing_params
                    )
                if (
                    args.quantum_actor
                    and not args.hybrid
                    and args.output_scaleing
                    # and not args.scheduled_output_scaleing
                ):
                    save_params.save_output_scaleing_params(args.chkpt_dir, output_scaleing_params)
                if args.quantum_critic:
                    save_params.save_critic_circuit_params(args.chkpt_dir, critic_layer_params)
                    save_params.save_critic_input_scaleing_params(
                        args.chkpt_dir, critic_input_scaleing_params
                    )
                save_params.save_state(args.chkpt_dir, global_step, next_obs, next_done)
                store_envs.save_envs(args.chkpt_dir, args.num_envs)

        else:  # if random baseline
            _, _, entropy_, _ = agent.get_random_action_and_value(
                get_act_dim(envs.single_action_space)
            )
            save_results.append_update_results(
                0,
                0,
                0,
                0,
                entropy_.item(),
                0,
                0,
                0,
                0,
                0,
                int(global_step / (time.time() - start_time)),
                0,
                global_step,
                args.gym_id,
                args.exp_name,
                args.circuit,
                args.seed,
            )

    # before exiting save everything and plot
    save_results.save_results()
    if not args.random_baseline:
        if not args.quantum_actor or not args.quantum_critic:
            agent.save_classical_network_params()
        if args.quantum_actor:
            save_params.save_actor_circuit_params(args.chkpt_dir, actor_layer_params)
            save_params.save_actor_input_scaleing_params(
                args.chkpt_dir, actor_input_scaleing_params
            )
        if (
            args.quantum_actor
            and not args.hybrid
            and args.output_scaleing
            # and not args.scheduled_output_scaleing
        ):
            save_params.save_output_scaleing_params(args.chkpt_dir, output_scaleing_params)
        if args.quantum_critic:
            save_params.save_critic_circuit_params(args.chkpt_dir, critic_layer_params)
            save_params.save_critic_input_scaleing_params(
                args.chkpt_dir, critic_input_scaleing_params
            )
        save_params.save_state(args.chkpt_dir, global_step, next_obs, next_done)
        store_envs.save_envs(args.chkpt_dir, args.num_envs)
        plot_training_results(
            args.results_dir,
            args.plot_dir,
            args.gym_id,
            args.exp_name,
            args.circuit,
            args.seed,
            args.batch_size,
            args.total_timesteps,
        )
    envs.close()
    writer.close()
