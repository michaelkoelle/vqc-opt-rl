from src.plot import plot_test_avg_final
from src.plot_grads import plot_gradient_avg
from src.plot_old import plot_training_results

results_dir = "qppo-slurm/results"
plots_dir = "plots"


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(3)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-3)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-1)-(67-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=1.0e-3", "Lr=1.0e-2", "Lr=5.0e-2", "Lr=1.0e-1", "Zufällige Aktion"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(4)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-3)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-1)-(88-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=1.0e-3", "Lr=1.0e-2", "Lr=5.0e-2)", "Lr=1.0e-1", "Zufällige Aktion"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(5)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-3)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-2)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=5.0e-2)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-1)-(109-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=1.0e-3", "Lr=1.0e-2", "Lr=5.0e-2)", "Lr=1.0e-1", "Zufällige Aktion"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(4,4)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-3)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-2)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=2.0e-2)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=5.0e-2)-(108-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=1.0e-3", "Lr=1.0e-2", "Lr=2.0e-2", "Lr=5.0e-2", "Zufällige Aktion"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(64,64)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    # "FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-3)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(5508-params)",
    #"FL-ppo-ac-NN(64,64)-(actor-lr=1.0e-2)-(5508-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=2.5e-4", "Lr=1.0e-3", "Lr=2.5e-3", "Lr=5.0e-3", "Zufällige Aktion"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0b-best-NN"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-2)-(109-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-2)-(108-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-3)-(5508-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 25000
alpha = 0.3
labels=["NN(3)-(67-Param)", "NN(4)-(88-Param)", "NN(5)-(109-Param)", "NN(4,4)-(108-Param)", "NN(64,64)-(5508-Param)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)








plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1a-qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr0.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=0.5e-3", "Lr=2.5e-3", "Lr=5.0e-3", "Lr=10.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1b-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-(72-params)",
    #"FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Critic-Lr(0.5e-4)", "QPPO-Critic-Lr(2.5e-4)", "QPPO-Critic-Lr(5.0e-4)", "QPPO-Critic-Lr(10.0e-4)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-exp-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht15000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht40000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht10000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht20000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht35000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=(10->0.1)e-3, HWZ=15000", "Lr=(10->0.1)e-3, HWZ=25000", "Lr=(10->0.1)e-3, HWZ=40000", "Lr=(10->1.0)e-3, HWZ=10000", "Lr=(10->1.0)e-3, HWZ=20000", "Lr=(10->1.0)e-3, HWZ=35000", "Zufällige Aktionsauswahl", "Konst-Lr=2.5e-3"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-2-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)-randominit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)-allsmallinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)-allmidinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)-allbiginit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)-gaussinit",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 50000
alpha = 0.3
labels=["Standard-Init", "Kleine-Init", "Mittlere-Init", "Große-Init", "Gauß-Init", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)







plot_dir = plots_dir + "/FL-actor-Test-1-output-scaleing"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    #"FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)", #"QPPO-globalesOutScale(5e-2)-(73-Param)"
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    #"FL-qppo-ac-simple_reuploading-exp_sced-hybrid_output-20params-(92-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 100000
alpha = 0.3
labels=["globalesOutScale(1e-3) (73-Param)", "globalesOutScale(1e-2) (73-Param)", "LokalesOutScale(1e-3) (76-Param)", "LokalesOutScale(1e-2) (76-Param)", "LokalesOutScale(5e-2) (76-Param)", "Zufällige Aktionsauswahl", "keinOutScale (72-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/FL-actor-Test-1-output-scaleing-2" #comparison with the classic PP0
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    #"FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)", #"QPPO-globalesOutScale(5e-2)-(73-Param)"
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 25000
alpha = 0.3
labels=["QPPO-globalesOutScale(1e-2)-(73-Param)", "QPPO-lokalesOutScale(1e-2)-(76-Param)", "QPPO-lokalesOutScale(5e-2)-(76-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-1-param"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-4-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-1-(73-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["globalesOutScale(1e-4)", "globalesOutScale(1e-3)", "globalesOutScale(1e-2)", "globalesOutScale(5e-2)", "globalesOutScale(1e-1)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-4-params"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-4-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-1-(76-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["LokalesOutScale(1e-4)", "LokalesOutScale(1e-3)", "LokalesOutScale(1e-2)", "LokalesOutScale(5e-2)", "LokalesOutScale(1e-1)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


"""
#we did some Tests in an alternate version of Frozen Lake, but werent able to produce any meaningful results in the available time

plot_dir = plots_dir + "/FL-actor-Test-1c-output-scaleing-in-alternate-Frozen-Lake-Environment"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
exp_names = [
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-2)-(108-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 100000
alpha = 0.3
labels=["QPPO-globalesOutScale(1e-2)-(73-Param)", "QPPO-lokalesOutScale(1e-2)-(76-Param)", "QPPO-lokalesOutScale(5e-2)-(76-Param)", "PPO(4,4)-Lr(1.0e-2)-(108-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

# Tests in alternate-Frozen-Lake-Environment discontinued
"""

plot_dir = plots_dir + "/FL-actor-Test-2-reuploading"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Data-Reuploading-(5e-3)", "Data-Reuploading-(1e-3)", "kein-Reuploading-(5e-3)", "kein-Reuploading-(1e-3)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
# two seeds fail for the higher OutScale lr without data rauploading



plot_dir = plots_dir + "/FL-actor-Test-2-alternate-circuit-architecture"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Hgog_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Jerbi-reuploading-no-input-scaleing-exp_sced-output_scaleing-1param-1e-2-9var_8enc_layers(73-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 50000
alpha = 0.3
labels=["QPPO-Standard-(73-Param)", "QPPO-HgogVQC-(73-Param)", "QPPO-JerbiVQC-(73-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Test-3-Ansatz-comparison"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-output_scaleing-1param-1e-3-(73-params)",               #lower OutScale lr (1e-3) since it fails for 5e-3
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-qlr-2.5e-3-output_scaleing-5e-3-(73-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-alle-Methoden-(73-Param)", "QPPO-ohne-Reuploading-(73-Param)", "QPPO-ohne-OutScale-(72-Param)", "QPPO-ohne-ExpLr-(73-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/FL-actor-Test-3b-Ansatz-comparison-global-outscale"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [

    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-4params-2.5e-2-(76-params)",
    #"FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-4params-1.25e-2-(76-params)",
    #"FL-qppo-ac-simple_reuploading-qlr-2.5e-3------------------output_scaleing-4params-2.5e-2-(76-params)",
    #"FL-qppo-ac-simple_reuploading-qlr-2.5e-3------------------output_scaleing-4params-1.25e-2-(76-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 25000
alpha = 0.3
labels=["QPPO-globalesOutScale-(76-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)






# Cartpole hyperparam tests


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(5,5)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-(actor-lr=5.0e-5)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=2.5e-4)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-3)-(67-params)",
    #"CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-2)-(67-params)",
    "CP-random-baseline-(0-params)",
    #"CP-ppo-ac-NN(5,5)-actorlr((25-5)e-5,ht=150000)-(67-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Lr=5.0e-5", "Lr=1.0e-4", "Lr=2.5e-4", "Lr=1.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(6,5)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,5)-(actor-lr=5.0e-5)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=2.5e-4)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-3)-(77-params)",
    #"CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-2)-(77-params)",
    "CP-random-baseline-(0-params)",
    #"CP-ppo-ac-NN(6,5)-actorlr((25-5)e-5,ht=150000)-(77-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Lr=5.0e-5", "Lr=1.0e-4", "Lr=2.5e-4", "Lr=1.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(6,6)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,6)-(actor-lr=5.0e-5)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=2.5e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-3)-(86-params)",
    #"CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-2)-(86-params)",
    "CP-random-baseline-(0-params)",
    #"CP-ppo-ac-NN(6,6)-actorlr((25-5)e-5,ht=150000)-(86-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Lr=5.0e-5", "Lr=1.0e-4", "Lr=2.5e-4", "Lr=1.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(7,7)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(7,7)-(actor-lr=5.0e-5)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-4)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=2.5e-4)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-3)-(107-params)",
    #"CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-2)-(107-params)",
    "CP-random-baseline-(0-params)",
    #"CP-ppo-ac-NN(7,7)-actorlr((25-5)e-5,ht=150000)-(107-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Lr=5.0e-5", "Lr=1.0e-4", "Lr=2.5e-4", "Lr=1.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(64,64)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=1.0e-4)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(4610-params)",
    #"CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(4610-params)",
    "CP-random-baseline-(0-params)",
    #"CP-ppo-ac-NN(64,64)-actorlr((10-2)e-5,ht=150000)-(4610-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Lr=5.0e-5", "Lr=1.0e-4", "Lr=2.5e-4", "Lr=1.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0b-best-NN"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-4)-(107-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(4610-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["NN(5,5)-(67-Param)", "NN(6,5)-(77-Param)", "NN(6,6)-(86-Param)", "NN(7,7)-(107-Param)", "NN(64,64)-(4610-Param)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0c-Lr-Scheduling"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-actorlr((25-5)e-5,ht=150000)-(67-params)",
    "CP-ppo-ac-NN(6,5)-actorlr((25-5)e-5,ht=150000)-(77-params)",
    "CP-ppo-ac-NN(6,6)-actorlr((25-5)e-5,ht=150000)-(86-params)",
    "CP-ppo-ac-NN(7,7)-actorlr((25-5)e-5,ht=150000)-(107-params)",
    "CP-ppo-ac-NN(64,64)-actorlr((10-2)e-5,ht=150000)-(4610-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["NN(5,5)-(67-Param)", "NN(6,5)-(77-Param)", "NN(6,6)-(86-Param)", "NN(7,7)-(107-Param)", "NN(64,64)-(4610-Param)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0d-Sigmoid-Lr-Scheduling"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-sigmoid-actorlr((25-5)e-5)-(67-params)",
    "CP-ppo-ac-NN(6,5)-sigmoid-actorlr((25-5)e-5)-(77-params)",
    "CP-ppo-ac-NN(6,6)-sigmoid-actorlr((25-5)e-5)-(86-params)",
    "CP-ppo-ac-NN(7,7)-sigmoid-actorlr((25-5)e-5)-(107-params)",
    "CP-ppo-ac-NN(64,64)-sigmoid-actorlr((10-2)e-5)-(4610-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["NN(5,5)-Sigmoid-Lr-(67-Param)", "NN(6,5)-Sigmoid-Lr-(77-Param)", "NN(6,6)-Sigmoid-Lr-(86-Param)", "NN(7,7)-Sigmoid-Lr-(107-Param)", "NN(64,64)-Sigmoid-Lr-(4610-Param)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0e-Scheduling-comp"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-actorlr((25-5)e-5,ht=150000)-(86-params)",
    "CP-ppo-ac-NN(6,6)-sigmoid-actorlr((25-5)e-5)-(86-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Fixe-Lr", "Exp-Lr", "Sigmoid-Lr", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)






plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1a-qlr-no_output_scale"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht15000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht35000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["QPPO Lr=2.5e-3", "QPPO Lr=1.0e-2", "QPPO Lr=(10->0.1)e-3, HWZ=15000", "QPPO Lr=(10->0.1)e-3, HWZ=25000", "QPPO Lr=(10->0.1)e-3, HWZ=35000",  "Zufällige Aktionsauswahl", "PPO(5,5) Lr=1.0e-4"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)




plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1b-qlr-no_output_scale2"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-2-(72-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["QPPO Lr=0.5e-3", "QPPO Lr=2.5e-3", "QPPO Lr=1.0e-2", "QPPO Lr=5.0e-2", "Zufällige Aktionsauswahl", "PPO(5,5) Lr=1.0e-4"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-2a-outscale-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-5-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-2-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-1-(73-params)",
    "CP-random-baseline-(0-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["OutScale(1e-5)", "OutScale(1e-4)", "OutScale(1e-3)", "OutScale(1e-2)", "OutScale(1e-1)", "Zufällige Aktionsauswahl", "keinOutScale"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-2b-outscale-lr-fixedlastentanglement"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-3-output_scaleing-1param-1e-3-(73-params)-fixedlastentanglement",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["QPPO-Lr(2.5e-3)-OutScale(1e-4)-Zirkuläre-LV", "QPPO-Lr(2.5e-3)-OutScale(1e-3)-Zirkuläre-LV", "QPPO-Lr(5.0e-3)-OutScale(1e-3)-Zirkuläre-LV", "QPPO-Lr(2.5e-3)-OutScale(1e-4)", "QPPO-Lr(2.5e-3)-OutScale(1e-3)", "QPPO-Lr(5.0e-3)-OutScale(1e-3)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
# for the tests before this one, circular entanglement was used for the last layer instead of onely entangeling the 1st and 3th Qubit plus the 2nd an 4th Qubit



plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3a-random-inits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",  # fully radom init
    #"CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-clippedrandominit",   #"Clipped-Init"
    #"CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allverysmall",        #"SehrKleine-Init"
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allmidinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allbiginit",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["Standard-Init", "Kleine-Init", "Mittlere-Init", "Große-Init", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3b-gaussinits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-unclipped_gaussinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-gaussinit",
    #"CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-rescaled_gaussinit", #"Kleine-Gauß-Init" #shows potential but couldnt be further explored
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["Standard-Init", "Gauß-Init", "Clipped-Gauß-Init", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3c-best-inits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-gaussinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Standard-Init", "Clipped-Gauß-Init", "Kleine-Init", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-4a-qlr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht_75000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht150000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05
labels=["Konst-Lr=0.5e-3", "Konst-Lr=2.5e-3", "Konst-Lr=10e-3", "Exp-Lr=(10->0.1)e-3, HWZ= 75000", "Exp-Lr=(2.5->0.1)e-3, HWZ=100000", "Exp-Lr=(2.5->0.1)e-3, HWZ=150000", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)




# input scaleing

plot_dir = plots_dir + "/CP-actor-Test-1a-maual-input-rescaleings"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit", # tanh(x)
    "CP-qppo-ac-simple_reuploading-tanh-x2-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # tanh(2x)
    "CP-qppo-ac-simple_reuploading-arctan-x1-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 2*arctan(x)
    "CP-qppo-ac-simple_reuploading-arctan-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 2*arctan(2x)
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["pi*tanh(x)-Reskalierung", "pi*tanh(2x)-Reskalierung", "2pi*arctan(x)-Reskalierung", "2pi*arctan(2x)-Reskalierung" , "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


"""
# we also tested 6 layer input scaling, but it performed poorly with the given hyperparameters (and we were out of time for another broad hyperparameter search)

plot_dir = plots_dir + "/CP-actor-Test-1b-input-scaleing"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht_80000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr1.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["QPPO(4-Layer)-InpScale-Lr=(2.5->0.1)e-3,HWZ=100000-(65-Param)", "QPPO(5-Layer)-InpScale-Lr=(2.5->0.1)e-3,HWZ=100000-(81-Param)", "QPPO(5-Layer)-InpScale-Lr=(2.5->0.1)e-3,HWZ=_80000-(81-Param)", "QPPO(6-Layer)-InpScale-Lr=(2.5->0.1)e-3,HWZ=100000-(97-Param)", "QPPO(6-Layer)-InpScale-Lr=(1.5->0.1)e-3,HWZ=100000-(97-Param)", "Zufällige Aktionsauswahl", "QPPO-manuelles-Rescale-Lr=(2.5->0.1)e-3,HWZ=100000-(73-Param)", "QPPO-globales-InpScale-Lr=(2.5->0.1)e-3,HWZ=100000-(77-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
"""

plot_dir = plots_dir + "/CP-actor-Test-1c-input-scaleing-500000-Steps"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.015
labels=["QPPO-manuelles-Rescale-(73-Param)", "QPPO-globales-InpScale-(77-Param)", "QPPO(5-Layer)-InpScale-(81-Param)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)", "PPO(6,6)-(86-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


"""
#(empirical) hyperparameter search:  we test some higher output scaling lrs, onely 2e-4 can archieve a higher Performance
#most runs werent completed and the results are hard to read and not very informative, therefore the plots are not included in the work

plot_dir = plots_dir + "/CP-actor-Test-2b-Ansatz-Comparison"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    #"CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.015
labels=["QPPO-manuelles-Rescale-(73-Param)(1e-3)", "QPPO-globales-InpScale-(77-Param)(1e-3)", "QPPO(4-Layer)-InpScale-(65-Param)(1e-3)", "QPPO(5-Layer)-InpScale-(81-Param)(1e-3)", "QPPO-manuelles-Rescale-(73-Param)(5e-4)", "QPPO-globales-InpScale-(77-Param)(5e-4)", "QPPO(4-Layer)-InpScale-(65-Param)(5e-4)", "QPPO(5-Layer)-InpScale-(81-Param)(5e-4)", "QPPO-manuelles-Rescale-(73-Param)(2e-4)", "QPPO-globales-InpScale-(77-Param)(2e-4)", "QPPO(4-Layer)-InpScale-(65-Param)(2e-4)", "QPPO(5-Layer)-InpScale-(81-Param)(2e-4)", "QPPO(6-Layer)-manuelles-Rescale-(73-Param)(1e-4)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

"""

plot_dir = plots_dir + "/CP-actor-Test-2-Ansatz-Comparison"
gym_id = "CartPole-v1"
exp_names = [
    #"CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(81-params)-5-layers",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    #"CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.015
labels=["QPPO-manuelles-Rescale-(73-Param)", "QPPO-globales-InpScale-(77-Param)", "QPPO(4-Layer)-InpScale-(65-Param)", "QPPO(5-Layer)-InpScale-(81-Param)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)






