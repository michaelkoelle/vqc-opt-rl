import math

import numpy as np

from args import parse_args

args = parse_args()


def make_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 2 - 1  # interval [-1, 1[
    layer_params = np.arctanh(layer_params)  # we use tanh weight remapping
    return layer_params


a = 0.01  # could be made a hyperparameter but is probaly uninteresting for tests


def make_clipped_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 2 - 1  # interval [-1, 1[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = np.clip(layer_params[i, j, k], 0 + a, 1 - a)
                else:
                    layer_params[i, j, k] = np.clip(layer_params[i, j, k], -1 + a, 0 - a)
                # interval [-0.99, -0.01] || [0.01, 0.99]
    layer_params = np.arctanh(layer_params)  # we use tanh weight remapping
    return layer_params


def make_all_toosmall_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 0.002 - 0.001  # interval [-0.001, 0.001[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + 0.0001
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - 0.0001
                # interval [-0.0011, -0.0001[ || [0.0001, 0.0011[
    layer_params = np.arctanh(layer_params)  # we use tanh weight remapping
    return layer_params


def make_all_verysmall_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 0.02 - 0.01  # interval [-0.01, 0.01[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + 0.001
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - 0.001
                # interval [-0.011, -0.001[ || [0.001, 0.011[
    layer_params = np.arctanh(layer_params)  # will later be inside a tanh function
    return layer_params


def make_all_small_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 0.2 - 0.1  # interval [-0.1, 0.1[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + a
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - a
                # interval [-0.11, -0.01[ || [0.01, 0.11[
    layer_params = np.arctanh(layer_params)  # will later be inside a tanh function
    return layer_params


def make_all_medium_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params - 0.5  # interval [-0.5, 0.5[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + 0.25
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - 0.25
                # interval [-0.75, -0.25[ || [0.25, 0.75[
    layer_params = np.arctanh(layer_params)  # will later be inside a tanh function
    return layer_params


def make_all_big_random_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.rand(n_qubits, n_layers, n_dims)
    layer_params = layer_params * 0.8 - 0.4  # interval [-0.4, 0.4[
    # clipping to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + 0.6 - a
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - 0.6 + a
                # interval [-0.99, -0.59[ || [0.59, 0.99[
    layer_params = np.arctanh(layer_params)  # we use tanh weight remapping
    return layer_params


def make_gauss_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.normal(0, 1, (n_qubits, n_layers, n_dims))
    # could be initialised at 0 so we clip it to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = np.clip(
                        layer_params[i, j, k], 0 + a, np.arctanh(1 - a)
                    )
                else:
                    layer_params[i, j, k] = np.clip(
                        layer_params[i, j, k], np.arctanh(-1 + a), 0 - a
                    )
    return layer_params


def make_unclipped_gauss_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.normal(0, 1, (n_qubits, n_layers, n_dims))
    return layer_params


def make_rescaled_gauss_init(n_qubits, n_layers, n_dims):
    layer_params = np.random.normal(0, 0.1, (n_qubits, n_layers, n_dims))
    # could be initialised at 0 so we clip it to avoid vanishing gradients
    for i in range(n_qubits):
        for j in range(n_layers):
            for k in range(n_dims):
                if layer_params[i, j, k] >= 0:
                    layer_params[i, j, k] = layer_params[i, j, k] + a
                else:
                    layer_params[i, j, k] = layer_params[i, j, k] - a
                # ~68% of values should lie in the interval [-0.11, -0.01[ & [0.01, 0.11[,
                # ~95% of values should lie in the interval [-0.21, -0.01[ & [0.01, 0.21[,
                # ~99% of values should lie in the interval [-0.31, -0.01[ & [0.01, 0.31[,
    return layer_params


def choose_init(n_qubits, n_layers, n_dims):
    if args.param_init == "random_clipped":
        layer_params = make_clipped_random_init(n_qubits, n_layers, n_dims)
        print("useing clipped random init")
    elif args.param_init == "gauss_distribution":
        layer_params = make_gauss_init(n_qubits, n_layers, n_dims)
        print("useing gauss init")
    elif args.param_init == "unclipped_gauss_distribution":
        layer_params = make_unclipped_gauss_init(n_qubits, n_layers, n_dims)
        print("useing gauss init")
    elif args.param_init == "rescaled_gauss_distribution":
        layer_params = make_rescaled_gauss_init(n_qubits, n_layers, n_dims)
        print("useing gauss init")
    elif args.param_init == "alltoosmall":
        layer_params = make_all_toosmall_random_init(n_qubits, n_layers, n_dims)
        print("useing all toosmall random init")
    elif args.param_init == "allverysmall":
        layer_params = make_all_verysmall_random_init(n_qubits, n_layers, n_dims)
        print("useing all verysmall random init")
    elif args.param_init == "allsmall":
        layer_params = make_all_small_random_init(n_qubits, n_layers, n_dims)
        print("useing all small random init")
    elif args.param_init == "allmid":
        layer_params = make_all_medium_random_init(n_qubits, n_layers, n_dims)
        print("useing all medium random init")
    elif args.param_init == "allbig":
        layer_params = make_all_big_random_init(n_qubits, n_layers, n_dims)
        print("useing all big random init")
    else:
        layer_params = make_random_init(n_qubits, n_layers, n_dims)
        print("useing default random init")
    return layer_params


def make_actor_layer_params():
    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones(0)

    elif args.quantum_actor and (args.circuit == "simple_reuploading_with_shared_input_scaleing"):
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones(args.n_qubits)
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            input_scaleing_params[0] = 1 / 4.8
            input_scaleing_params[2] = 1 / 0.418

    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones((args.n_qubits, args.n_var_layers))
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            # input_scaleing_params = np.full((args.n_qubits, args.n_var_layers), 2)
            for j in range(args.n_var_layers):
                input_scaleing_params[0][j] = 1 / 4.8
                input_scaleing_params[2][j] = 1 / 0.418

    elif args.quantum_actor and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 2)
        input_scaleing_params = np.ones(0)

    elif args.quantum_actor and args.circuit == "Jerbi-reuploading":
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 2)
        input_scaleing_params = np.ones((args.n_qubits, args.n_var_layers - 1, 2))
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            # input_scaleing_params = np.full((args.n_qubits, args.n_var_layers - 1, 2), 2)
            for j in range(args.n_var_layers):
                for i in range(2):
                    input_scaleing_params[0][j][i] = 1 / 4.8
                    input_scaleing_params[2][j][i] = 1 / 0.418

    else:
        layer_params = np.random.rand(0)
        input_scaleing_params = np.ones(0)
        print("no actor_layer_params needed")
    return layer_params, input_scaleing_params


def make_critic_layer_params():
    if args.quantum_actor and (
        args.circuit == "simple"
        or args.circuit == "Hgog"
        or args.circuit == "simple_reuploading"
        or args.circuit == "Hgog_reuploading"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones(0)

    elif args.quantum_actor and (args.circuit == "simple_reuploading_with_shared_input_scaleing"):
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones(args.n_qubits)
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            input_scaleing_params[0] = 1 / 4.8
            # input_scaleing_params[1]= 2
            input_scaleing_params[2] = 1 / 0.418
            # input_scaleing_params[3]= 2

    elif args.quantum_actor and (
        args.circuit == "simple_reuploading_with_input_scaleing"
        or args.circuit == "Hgog_reuploading_with_input_scaleing"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 3)
        input_scaleing_params = np.ones((args.n_qubits, args.n_var_layers))
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            # input_scaleing_params = np.full((args.n_qubits, args.n_var_layers), 2)
            for j in range(args.n_var_layers):
                input_scaleing_params[0][j] = 1 / 4.8
                input_scaleing_params[2][j] = 1 / 0.418

    elif args.quantum_critic and (
        args.circuit == "Jerbi-no-reuploading-no-input-scaleing"
        or args.circuit == "Jerbi-reuploading-no-input-scaleing"
    ):
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 2)
        input_scaleing_params = np.ones(0)

    elif args.quantum_critic and args.circuit == "Jerbi-reuploading":
        # choose parameter initialisation method
        layer_params = choose_init(args.n_qubits, args.n_var_layers, 2)
        input_scaleing_params = np.ones((args.n_qubits, args.n_var_layers - 1, 2))
        if args.gym_id == "CartPole-v0" or args.gym_id == "CartPole-v1":
            # input_scaleing_params = np.full((args.n_qubits, args.n_var_layers - 1, 2), 2)
            for j in range(args.n_var_layers):
                for i in range(2):
                    input_scaleing_params[0][j][i] = 1 / 4.8
                    input_scaleing_params[2][j][i] = 1 / 0.418

    else:
        layer_params = np.random.rand(0)
        input_scaleing_params = np.ones(0)
        print("no critic_layer_params needed")
    return layer_params, input_scaleing_params
