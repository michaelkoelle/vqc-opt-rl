**"Optimierung von Variational Quantum Circuits für Hybride Quantum Proximal Policy Optimization Algorithmen" von Timo Witter** 
=
**Dieses Git-Repository unfasst die folgenden Dateien und Verzeichnisse:**

- jobs:
    - job.sh
        >Wird für "run-jobs.sh" benötigt und erstellt eine pyenv für die Tests
    - run-jobs.sh           
        >Dient zum Ausführen der slurm-Aufrufe, mit denen die Algorithmen auf den Rechnern des LRZ ausgeführt wurden

- plots
    >Enthält die mithilfe von "plot_log.py" erstellten Plots, welche in der Arbeit verwendet wurden

- qppo-slurm
    >Enthält die Checkpoints und Ergebnisse aller durchgeführten runs

- src
    >Enthält die Python files des Algorihmus:
    - agent.py
        >Logik des Agenten (Actor und Critic) und er Nauronalen Netzwerke
    - args.py
        >Übergeben der hyperparameter
    - calc_num_param.py
        >berechnet die Anzahl der Parameter des Actor und Critic zu rein informativen zwecken
    - circuits.py
        >Implementierung der verschiedenen Schaltkreise
    - env.setup.py
        >Initialisierungsfunktion der einzelnen Umgebungen für die multi-vector-environment
    - envs_storage.py
        >Funktionen zur Speicherung von Umgebungsdaten für ein eventuelles Fortsetzen des Lernprozees
    - layer_params.py
        >Initialisierung der Parameter der Schaltkreise
    - main.py
        >Ausführbare Datei mit der Kernlogik des (Quantum) Proximal Policy Optimization Algorithmus
    - plot_grads.py
        >Funktionen für das Plotten von Gradienten
    - plot_old.py
        >Alte Plotfunktionen, welche für einen simplen abschließenden Plot in "main.py" eingesetzt werden
    - plot.py
        >Überarbeitete Plotfunktionen, welche für die Auswertung der Testergebnisse verwendet wurden
    - save_params.py
        >Funktionen zum Speichern der Parameter des Schaltkreises (Speichern der NN Parameter in "agent.py")
    - save_results.py
        >Funktionen zu abspeichern der Ergebnisse
    - ShortestPathFrozenLake.py
        >Veränderte Frozen Lake Umgebung mit dense Reward
    - transform_funks.py
        >Normalisierungs- und Kodierungsfunktionen für den Schaltkreis
    - utils.py
        >Funktionen zur Bestimmung der Dimensionalitäten der Umgebung

- plot_log.py
    >Ausführbare Datei mit allen Funktionsaufrufen, die zur Erstellung der Plots verwendet wurden

- run_log.txt
    >Auflistung aller Slurm-Aufrufe, die (innerhalb von "run-jobs.sh") eingesetzt wurden um die Testergebnisse zu produzieren




**Ausführen des Algorithmus:**
>Um den PPO in einer Umgebung des OpenAI Gym (wie Cart Pole) trainieren zu lassen muss z.B. ein `$ python src/main.py --gym-id CartPole-v1` Aufruf ausgeführt werden. Um mehrere Seeds parallel über Slurm durchzuführen werden stattdessen in "jobs/run-jobs.sh" mehrere Aufrufe von "jobs/jobs.sh" (mit den entsprechenden argparse Argumenten) hinterlegt (und mit `$ jobs/run-jobs.sh` gestartet).

**Hyperparameter Einstellungen mit argparse**
>Es gibt eine vielzahl an weiteren Hyperparametern, welche in der Kommandozeile übergeben werden können. Diese können mit dem `python src/main.py --help` Befehl oder in "src/args.py" eingesehen werden. Da die dies schnell unübersichtlich wird haben wir in "run_log.txt" alle von uns (in "jobs/run-jobs.sh") ausgeführten slurm-commands aufgelistet.

**Requirements:**
>Die in "requirements.txt" aufgeführten Requirements sollten beim Ausführen von `jobs/jobs.sh` automatisch installiert werden. Der Algorithmus wurde nur für Cart Pole und Frozen Lake getestet und braucht für andere Umgebungen vermutlich Anpassungen.

**Checkpoints:**
>Der Algorithmus speichert in festen Intervallen alle Ergebnisse und Parameter mit deren Hilfe ein Durchlauf auch nach Ablauf der angegebenen Umgebungsschritte oder nach einem Absturz fortgesetzt werden kann. Für Frozen Lake war es ebenfalls möglich den Zustand der Multi Vector Environment wiederherzustellen, für Cart Pole ist uns das leider nicht gelungen und wir mussten nach jeder Wiederherstellung die Umgebungen neu initialisieren. Um einen existierenden Checkpoint zu laden muss der exakt selbe Funktionsaufruf erneut ausgeführt werden, nur dass `--load-chkpt` auf `True` gesetzt werden muss und für `--total-timesteps` ein größerer Wert eingesetzt werden kann.

>**Verwendung der Checkpoints in unserer Arbeit und der Einfluss auf die Reproduzierbarkeit:**
>>Obwohl es nicht perfekt ist, war das Checkpointsystem für unsere Tests in Cart Pole unumgänglich, da die Durchläufe (bei Verwendung des Actor VQC) in der Regel etwa 3 Tage duerten. Die meisten Runs in dieser Umgebung wurden (nur bei Verwendung des Actor VQC) erst einmal für 150000 Zeitschritte angetestet und danach vom Checkpoint aus auf 500000 Schritte ausgeweitet. Im Falle von Node-Fails des Slurm Systems wurde der Verlauf ebenfalls vom letzten Checkpoint aus fortgesetzt, um den Lernfortschritt nicht zu verlieren. Die SPS (Steps per Second) Plots sollten einen guten überblick darüber schaffen, wann neugestartet wurde, da bei jedem Neustart ein Spike auftritt. Empirisch gesehen sollte der Einfluss eines solchen Neustarts auf die Durchschnittliche Leistung des (Quantum) PPOs vernachlässigbar sein, da der Algorihmus seine Lernzyklen ohnehin voneinander unabhängig in festen Zeitintervallen ausführt und nur "On Policy" Daten (aus dem letzten Lernintervall) verwenden und somit kein Lernfortschritt verloren geht. Der einzige Unterschied besteht daher darin, dass der Zufallsgenerator resetted wird und die im schlimmsten Fall die Endphase eines vielversprechenden Durchlaufs (für jede der n Parrallelen Umgebungen) nicht beobachtet werden kann. Da über die Dauer von 500000 Zeitschritten mindestens 1000 (im Durchschnitt ungefähr 1500 bis 2000) Episoden durchlaufen werden sollte der Einfluss des Resetts also relativ gering sein. Im Hinblick auf die Reproduzierbarkeit wollen wir dies jedoch trotzdem erwähnt haben.

**Plots**
>Um die Testergebnisse zu Plotten wird `src.plot.plot_test_avg_final("qppo-slurm/results", "plots", gym_id, exp_names, seeds, alpha, max_steps, labels)` ausgeführt, wobei gilt:
>- `gym_id` ist die id der Gym Umgebung des Tests.
>- `exp_name` ist eine Liste mit den Namen der Experimente, die geplotted werden sollen. 
>- `seeds` ist eine Liste mit den Seeds die eingesetzt wurden.
>- `alpha` ist der alpha wert für das ewm (exponentially weighted moveing average).
>- `max_steps` ist die Azahl der Zeitschritte über die geplotted werden soll
>- `labels` ist eine Liste mit Namenskürzeln, die für die Legende verwendet werden sollen (in der selben Reihenfolge wie `exp_name`).
>In "plot_log.py" haben wir alle von uns verwendeten aufrufe der Funktion aufgelistet.

