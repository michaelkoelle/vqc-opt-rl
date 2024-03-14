import os

import pandas as pd
import torch.nn as nn


class Save_results(nn.Module):
    def __init__(self, results_dir, load_chkpt, record_grads):
        super(Save_results, self).__init__()
        self.results_dir = results_dir
        self.df_episode_exists = False
        self.df_update_exists = False
        self.df_gradient_exists = False
        file_pathExists = os.path.exists(self.results_dir)
        if not file_pathExists:
            os.makedirs(self.results_dir)
        if load_chkpt:
            df_pathExists = os.path.exists(self.results_dir + "/episode_results.csv")
            df_pathExists2 = os.path.exists(self.results_dir + "/update_results.csv")
            if not df_pathExists:
                print("ERROR episode_results.csv can not be found")
            elif not df_pathExists2:
                print("ERROR update_results.csv can not be found")
            else:
                self.df_episode = pd.read_csv(self.results_dir + "/episode_results.csv")
                self.df_update = pd.read_csv(self.results_dir + "/update_results.csv")
                self.df_episode_exists = True
                self.df_update_exists = True
                if record_grads:
                    df_pathExists3 = os.path.exists(self.results_dir + "/gradient_results.csv")
                    if not df_pathExists3:
                        print("ERROR gradient_results.csv can not be found")
                    self.df_gradient = pd.read_csv(self.results_dir + "/gradient_results.csv")
                    self.df_gradient_exists = True

    def append_episode_results(
        self, episode_reward, episode_length, global_step, gym_id, exp_name, circuit, seed
    ):
        episode_results = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "global_step": global_step,
            "gym_id": gym_id,
            "exp_name": exp_name,
            "circuit": circuit,
            "seed": seed,
        }

        df = pd.DataFrame(data=episode_results, index=[0])

        if not self.df_episode_exists:
            self.df_episode = df
            self.df_episode_exists = True
        else:
            self.df_episode = pd.concat([self.df_episode, df], ignore_index=True)

    def append_update_results(
        self,
        learning_rate,
        qlearning_rate,
        value_loss,
        policy_loss,
        entropy,
        loss,
        old_approx_kl,
        approx_kl,
        clipfrac,
        explained_variance,
        SPS,
        output_scaleing,
        global_step,
        gym_id,
        exp_name,
        circuit,
        seed,
    ):
        update_results = {
            "learning_rate": learning_rate,
            "qlearning_rate": qlearning_rate,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "loss": loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "explained_variance": explained_variance,
            "SPS": SPS,
            "output_scaleing": output_scaleing,
            "global_step": global_step,
            "gym_id": gym_id,
            "exp_name": exp_name,
            "circuit": circuit,
            "seed": seed,
        }

        df = pd.DataFrame(data=update_results, index=[0])
        if not self.df_update_exists:
            self.df_update = df
            self.df_update_exists = True
        else:
            self.df_update = pd.concat([self.df_update, df], ignore_index=True)

    def append_gradient_results(
        self,
        actor_grads_abs_mean,
        actor_gradients_var,
        actor_gradients_std,
        critic_grads_abs_mean,
        critic_gradients_var,
        critic_gradients_std,
        global_step,
        gym_id,
        exp_name,
        circuit,
        seed,
    ):
        gradient_results = {
            "actor_grads_abs_mean": actor_grads_abs_mean,
            "actor_gradients_var": actor_gradients_var,
            "actor_gradients_std": actor_gradients_std,
            "critic_grads_abs_mean": critic_grads_abs_mean,
            "critic_gradients_var": critic_gradients_var,
            "critic_gradients_std": critic_gradients_std,
            "global_step": global_step,
            "gym_id": gym_id,
            "exp_name": exp_name,
            "circuit": circuit,
            "seed": seed,
        }

        df = pd.DataFrame(data=gradient_results, index=[0])
        if not self.df_gradient_exists:
            self.df_gradient = df
            self.df_gradient_exists = True
        else:
            self.df_gradient = pd.concat([self.df_gradient, df], ignore_index=True)

    def save_results(self):
        if self.df_episode_exists:
            self.df_episode.to_csv(
                self.results_dir + "/episode_results.csv", sep=",", encoding="utf-8", index=False
            )
        if self.df_update_exists:
            self.df_update.to_csv(
                self.results_dir + "/update_results.csv", sep=",", encoding="utf-8", index=False
            )
        if self.df_gradient_exists:
            self.df_gradient.to_csv(
                self.results_dir + "/gradient_results.csv", sep=",", encoding="utf-8", index=False
            )
