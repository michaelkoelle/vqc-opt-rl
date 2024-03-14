import os

import numpy as np
import torch.nn as nn


class Store_envs(nn.Module):
    def __init__(self):
        super(Store_envs, self).__init__()
        self.storage_0 = np.array([])
        self.storage_1 = np.array([])
        self.storage_2 = np.array([])
        self.storage_3 = np.array([])
        self.restore_envs = np.array([True, True, True, True])

    def get_storage(self, i):
        if i == 0:
            return self.storage_0
        elif i == 1:
            return self.storage_1
        elif i == 2:
            return self.storage_2
        elif i == 3:
            return self.storage_3
        else:
            print("Storage ERROR")
            raise NotImplementedError()

    def set_storage(self, i, x):
        if i == 0:
            self.storage_0 = x
        elif i == 1:
            self.storage_1 = x
        elif i == 2:
            self.storage_2 = x
        elif i == 3:
            self.storage_3 = x
        else:
            print("Storage ERROR")
            raise NotImplementedError()

    def get_storage_file(self, i, chkpt_dir):
        if i == 0:
            return os.path.join(chkpt_dir, "envs_0.txt")
        elif i == 1:
            return os.path.join(chkpt_dir, "envs_1.txt")
        elif i == 2:
            return os.path.join(chkpt_dir, "envs_2.txt")
        elif i == 3:
            return os.path.join(chkpt_dir, "envs_3.txt")
        else:
            print("Storage ERROR")
            raise NotImplementedError()

    def append_storage(self, i, x):
        self.set_storage(i, np.append(self.get_storage(i), x))

    def store_envs(self, actions, dones, num_steps, num_envs):
        action = actions.cpu().detach().numpy()
        done = dones
        for j in range(num_steps):
            for i in range(num_envs):
                if done[j, i] == 1.0:
                    self.set_storage(i, [action[j, i]])
                else:
                    self.append_storage(i, action[j, i])

    def save_envs(self, chkpt_dir, num_envs):
        for i in range(num_envs):
            np.savetxt(
                self.get_storage_file(i, chkpt_dir), self.get_storage(i), delimiter=" ", fmt="%i"
            )

    def load_envs(self, chkpt_dir, num_envs):
        for i in range(num_envs):
            self.set_storage(i, np.loadtxt(self.get_storage_file(i, chkpt_dir), delimiter=" "))
