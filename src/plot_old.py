import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from matplotlib import rc

mpl.use("Agg")
sns.set_theme()


# plot results at the end of the run
def plot_training_results(
    results_dir, plot_dir, gym_id, exp_name, circuit, seed, batchsize, max_steps
):
    pathExists = os.path.exists(plot_dir)
    if not pathExists:
        os.makedirs(plot_dir)

    episode_results_dir = os.path.join(results_dir, "episode_results.csv")
    episode_results = pd.read_csv(episode_results_dir)

    update_results_dir = os.path.join(results_dir, "update_results.csv")
    update_results = pd.read_csv(update_results_dir)

    avg_episode_results = avg_episode_results_of_update(
        episode_results, gym_id, exp_name, seed, batchsize, max_steps
    )

    # plotting over update
    plot_avg_episode_reward_by_seed(avg_episode_results, plot_dir)
    plot_avg_episode_length_by_seed(avg_episode_results, plot_dir)

    plot_learning_rate_by_seed(update_results, plot_dir)
    plot_qlearning_rate_by_seed(update_results, plot_dir)
    plot_value_loss_by_seed(update_results, plot_dir)
    plot_policy_loss_by_seed(update_results, plot_dir)
    plot_entropy_by_seed(update_results, plot_dir)
    plot_loss_by_seed(update_results, plot_dir)
    plot_old_approx_kl_by_seed(update_results, plot_dir)
    plot_approx_kl_by_seed(update_results, plot_dir)
    plot_clipfrac_by_seed(update_results, plot_dir)
    plot_explained_variance_by_seed(update_results, plot_dir)
    plot_SPS_by_seed(update_results, plot_dir)
    plot_output_scaleing_by_seed(update_results, plot_dir)


def avg_episode_results_of_update(
    episode_results_df, gym_id, exp_name, seed, batchsize, max_steps
):
    # make avg over update for plotting
    avg_rewards = []
    avg_lengths = []
    avg_global_step = []
    avg_gym_id = []
    avg_exp_name = []
    # avg_circuit = []
    avg_seed = []

    for i in range(batchsize, max_steps, batchsize):
        df_tmp = episode_results_df[
            (episode_results_df.global_step <= i)
            & (episode_results_df.global_step > (i - batchsize))
        ]
        avg_rewards.append(df_tmp["episode_reward"].mean())
        avg_lengths.append(df_tmp["episode_length"].mean())
        avg_global_step.append(i)
        avg_gym_id.append(gym_id)
        avg_exp_name.append(exp_name)
        # avg_circuit.append(circuit)
        avg_seed.append(seed)

    avg_results = {
        "avg_rewards": avg_rewards,
        "avg_lengths": avg_lengths,
        "global_step": avg_global_step,
        "gym_id": gym_id,
        "exp_name": avg_exp_name,
        #'circuit': circuit,
        "seed": avg_seed,
    }

    return pd.DataFrame(data=avg_results)


def plot_avg_episode_reward_by_seed(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_rewards",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "episode_reward_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_reward_by_exp_name(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_rewards",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "episode_reward_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_seed(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_lengths",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "episode_lenght_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_avg_episode_length_by_exp_name(episode_results, dir):
    sns.relplot(
        data=episode_results,
        kind="line",
        x="global_step",
        y="avg_lengths",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "episode_lenght_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "learning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_learning_rate_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="learning_rate",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "learning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "qlearning_rate_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_qlearning_rate_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="qlearning_rate",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "qlearning_rate_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "value_loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_value_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="value_loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "value_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "policy_loss_by.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_policy_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="policy_loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "policy_loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "entropy_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_entropy_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="entropy",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "entropy_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "loss_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_loss_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="loss",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "loss_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_old_approx_kl_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="old_approx_kl",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "old_approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_approx_kl_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="approx_kl",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "approx_kl_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_clipfrac_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="clipfrac",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "clipfrac_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_explained_variance_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="explained_variance",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "explained_variance_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_SPS_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="SPS",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "SPS_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_SPS_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="SPS",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "SPS_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_seed(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        col="exp_name",
        hue="seed",
        # style="seed",
    )
    plot_dir = os.path.join(dir, "output_scaleing_by_seed.png")
    plt.savefig(plot_dir)
    plt.close()


def plot_output_scaleing_by_exp_name(update_results, dir):
    sns.relplot(
        data=update_results,
        kind="line",
        x="global_step",
        y="output_scaleing",
        col="gym_id",
        hue="exp_name",
        # style="exp_name",
    )
    plot_dir = os.path.join(dir, "output_scaleing_by_exp_name.png")
    plt.savefig(plot_dir)
    plt.close()
