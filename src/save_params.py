import os

import torch


def save_actor_circuit_params(chkpt_dir, actor_params):
        actor_checkpoint_file = os.path.join(chkpt_dir, 'actor_circuit_params')
        torch.save(actor_params, actor_checkpoint_file)

def save_actor_input_scaleing_params(chkpt_dir, actor_input_scaleing_params):
        actor_scale_checkpoint_file = os.path.join(chkpt_dir, 'actor_input_scaleing_params')
        torch.save(actor_input_scaleing_params, actor_scale_checkpoint_file)

def save_critic_circuit_params(chkpt_dir, critic_params):
        critic_checkpoint_file = os.path.join(chkpt_dir, 'critic_circuit_params')
        torch.save(critic_params, critic_checkpoint_file)

def save_critic_input_scaleing_params(chkpt_dir, critic_input_scaleing_params):
        critic_scale_checkpoint_file = os.path.join(chkpt_dir, 'critic_input_scaleing_params')
        torch.save(critic_input_scaleing_params, critic_scale_checkpoint_file)

def save_output_scaleing_params(chkpt_dir, output_scaleing_params):
        output_scaleing_checkpoint_file = os.path.join(chkpt_dir, 'output_scaleing_params')
        torch.save(output_scaleing_params, output_scaleing_checkpoint_file)


    
def load_actor_circuit_params(chkpt_dir):
        actor_checkpoint_file = os.path.join(chkpt_dir, 'actor_circuit_params')
        actor_params = torch.load(actor_checkpoint_file)
        return actor_params

def load_actor_input_scaleing_params(chkpt_dir):
        actor_input_checkpoint_file = os.path.join(chkpt_dir, 'actor_input_scaleing_params')
        actor_input_scaleing_params = torch.load(actor_input_checkpoint_file)
        return actor_input_scaleing_params
    
def load_critic_circuit_params(chkpt_dir):
        critic_checkpoint_file = os.path.join(chkpt_dir, 'critic_circuit_params')
        critic_params = torch.load(critic_checkpoint_file)
        return critic_params

def load_critic_input_scaleing_params(chkpt_dir):
        critic_input_checkpoint_file = os.path.join(chkpt_dir, 'critic_input_scaleing_params')
        critic_input_scaleing_params = torch.load(critic_input_checkpoint_file)
        return critic_input_scaleing_params

def load_output_scaleing_params(chkpt_dir):
        output_scaleing_checkpoint_file = os.path.join(chkpt_dir, 'output_scaleing_params')
        output_scaleing_params = torch.load(output_scaleing_checkpoint_file)
        return output_scaleing_params



def save_state(chkpt_dir, global_step, next_obs, next_done):
    checkpoint_file = os.path.join(chkpt_dir, 'state')
    db = {'global_step': torch.tensor(global_step), 'next_obs': next_obs, 'next_done': next_done}
    torch.save(db, checkpoint_file)

def load_state(chkpt_dir):
    checkpoint_file = os.path.join(chkpt_dir, 'state')
    loaded = torch.load(checkpoint_file)
    return int(loaded['global_step'].item()), loaded['next_obs'], loaded['next_done']



















