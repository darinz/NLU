# 02. Running NLP Experiments

## Introduction

Running NLP experiments involves implementing, executing, and monitoring experiments to ensure reliable and reproducible results. This guide covers best practices for experiment setup, automation, logging, and troubleshooting.

## 1. Experiment Setup
- Organize code, data, and configuration files in a clear directory structure.
- Use configuration files (YAML, JSON) to manage hyperparameters and settings.

### Example Directory Structure
```
project/
  data/
  src/
  configs/
  results/
  logs/
```

## 2. Automation and Experiment Management
- Use experiment management tools (e.g., MLflow, Sacred, Weights & Biases) to track runs and results.
- Automate repetitive tasks (training, evaluation, logging) with scripts or workflow managers (e.g., Snakemake, Make).

### Python Example: Logging with MLflow
```python
import mlflow
mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", 0.92)
mlflow.end_run()
```

## 3. Hyperparameter Tuning
- Systematically explore hyperparameters using grid search, random search, or Bayesian optimization.

### Mathematical Formulation: Grid Search
Given hyperparameter sets $`H_1, H_2, ..., H_n`$:
```math
\text{Best} = \arg\max_{h \in H_1 \times ... \times H_n} \text{Score}(h)
```

### Python Example: Grid Search with Scikit-learn
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {"n_estimators": [10, 50], "max_depth": [2, 4]}
clf = RandomForestClassifier()
grid = GridSearchCV(clf, param_grid, cv=3)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

## 4. Logging and Monitoring
- Log all experiment details: parameters, metrics, code versions, and environment info.
- Monitor resource usage (CPU, GPU, memory) and experiment progress.

## 5. Troubleshooting and Debugging
- Use assertions and unit tests to catch errors early.
- Visualize training curves and metrics to spot issues (e.g., overfitting, vanishing gradients).

### Python Example: Plotting Training Curves
```python
import matplotlib.pyplot as plt
train_loss = [0.9, 0.7, 0.5, 0.4]
val_loss = [1.0, 0.8, 0.6, 0.5]
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curves')
plt.show()
```

## Summary
- Running NLP experiments requires careful setup, automation, logging, and monitoring.
- Use experiment management tools and systematic hyperparameter tuning for reliable results.
- Troubleshooting and visualization are key for diagnosing and resolving issues. 