# **Classical-AutoML**
An AutoML framework for classical machine learning algorithms, automating model selection and hyperparameter tuning through Bayesian optimization, portfolio-based meta-learning, and multi-fidelity evaluation using Successive Halving.

## **🚀Highlights**

- 🔍 **Model and Hyperparameter Selection**: Automatically selects the best model and configuration for a given dataset using SMBO (Sequential Model-Based Optimization).
- 🧠 **Meta-Learning with Portfolio Construction**: Warm-starts the optimization using a greedy portfolio of configurations collected across datasets.
- ⚖️ **Multi-Fidelity Evaluation**: Uses Successive Halving to allocate more resources to promising candidates and reduce evaluation cost.
- ⚙️ **Focus on Classical ML Models**: Supports a curated set of efficient, non-deep-learning models — including Random Forests, Extra Trees, MLP, Stochastic Gradient Descent, Passive Aggressive, and Histogram-based Gradient Boosting.
- 🖧 **Parallelized Meta-Learning and Benchmarking with Ray**: Ray is used to parallelize large-scale evaluation of configurations across datasets both during portfolio construction (meta-learning) and during benchmarking runs significantly speeding up experimentation.
- 🤝 **Competitive Benchmarking**: Evaluated against Auto-sklearn 2.0 and achieves competitive accuracy across test datasets.

## ⚙️ Ray Cluster Setup & Dashboard Access

This project uses [Ray](https://docs.ray.io/) to parallelize configuration evaluations during meta-learning and benchmarking. 

To simplify setup, a `scripts/start_cluster.sh` script is provided to automatically:
- Start a Ray head node
- Connect and start Ray worker nodes via SSH
- Activate virtual environments and configure `PYTHONPATH` on all nodes

📌 Before running the script, ensure passwordless SSH access from the head node to all worker nodes.

📌 You must customize the `VENV_PATH`, `PROJECT_PATH`, `HEAD_IP`, and `WORKER_NODES` variables in the script based on your system and cluster configuration.

### 🔐 SSH Setup (One-Time)

On the head node:
```bash
ssh-keygen                         # Press Enter to accept default
ssh-copy-id username@<worker_ip>   # Repeat for each worker IP
```

Once started, the Ray dashboard is available at:
``` bash
http://<HEAD_IP>:8265
```

## **▶️Running the Project**

### **Requirements**:
The `requirements.txt` file contains all the necessary Python dependencies for the project. Place any additional required dependencies here. You can install them using:
``` bash
    pip install -r requirements.txt
```


To use the AutoML framework on your own dataset:

### 1. **Import and Initialize**
```python
from backend.autoclassifier import AutoClassifier

clf = AutoClassifier(
    seed=42,
    walltime_limit=300,   # seconds
    min_budget=10,
    max_budget=200
)
```

### 2. **Fit the model on training data**
```python
clf.fit(X_train, y_train)
```
Make sure your dataset is in pandas.DataFrame. The framework internally handles categorical encoding and scaling.

### 3. **Predict on test data**
```python
y_pred = clf.predict(X_test)
```

### 4. **Best configuration**
```python
best_config = clf.best_config
```

## **🗂️Project Structure**
```bash
    backend
    │   autoclassifier.py
    │   autosklearn_benchmark.py
    │   benchmark.py
    │   config.json
    │   config.py
    │   Optimizer.py
    │   plots.py
    │   test.py
    │   
    ├───meta_learning
    │       best_candidate_run.py
    │       candidates.py
    │       candidates._openml.py
    │       performance_matrix.py
    │       portfolio.py
    │       preprocess.py
    │
    └───pipelines
            BasePipeline.py
            BuildConfigurations.py
            ExtraTreesPipeline.py
            HistGradientBoostingPipeline.py
            MLPPipeline.py
            PassiveAggressivePipeline.py
            PipelineRegistry.py
            RandomForestPipeline.py
            SgdPipeline.py
            util.py
```

## **📊Results**
The framework was evaluated on benchmark datasets from OpenML under a strict 10-minute runtime constraint per task. The experiments focused on two aspects: the impact of meta-learning and the comparison against Auto-sklearn 2.0.

### 🔍 Meta-Learning Impact
- On **~80% of datasets**, meta-learning either improved test accuracy or matched baseline performance (within 1% difference).
- This demonstrates that portfolio-based initialization is especially effective under tight budget constraints.

### 🆚 Comparison with Auto-sklearn 2.0
- The framework **outperformed Auto-sklearn on ~50% of the datasets**.
- On another **~30% of datasets**, performance differed by less than 1%.
- In rare cases of lower accuracy, it was due to training timeouts on large datasets.

### ⚙️ Experimental Setup
- Max runtime: **10 minutes per task**
- Budgets: `min_budget = 10`, `max_budget = 500`
- Experiments parallelized using **Ray** across 7 machines (164 CPUs)

📄 **[Read the full report](./report/automl_report.pdf)** for methodology, configuration details, and accuracy plots.
