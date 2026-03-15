<div align="center">
    <a href="https://www.michiganinvestmentgroup.com/"><img src="./media/logo.jpeg"></a>
</div>

# MIG Algo Comp

- [Competition Docs](https://mig-algo-challenge.vercel.app/)
- [Discord](https://discord.gg/depR4xR2)

## Sections
[About This Repo](#about)

[Getting Started](#getting-started)

[Fundamental Prep](#fundamental-prep)

[Developer Environments](#developer-environments)

[Machine Learning & Local Training](#machine-learning)


## <a name="about"></a>About This Repo
The repo will give you everything you need to get started building Algorithms for the MIG Algo Competition.

## <a name="getting-started"></a>Getting Started

1. [Create a Python virtual environment](#developer-environments) for the examples and future algorithm development
2. [Git clone](#cloning-this-repo) this repo onto your computer and go through the examples and make sure you understand
3. Brush up on your [python/git skills](#fundamental-prep)

## <a name="developer-environments"></a>Developer Environments

We will be using Python's built-in `venv` module to create isolated Python environments that are consistent across contestant's machines. Make sure you have Python 3 installed — you can check with `python3 --version`.

Create a new virtual environment (where `<env_name>` is the name of your environment, e.g. `migenv`):
```
$ python3 -m venv <env_name>
```

Activate the virtual environment:

**macOS/Linux:**
```
$ source <env_name>/bin/activate
```

**Windows:**
```
$ <env_name>\Scripts\activate
```

You should now see the name of the environment in your terminal like so:
```
(<env_name>) user@computer % ...
```

Next, install the Python packages that will be used when developing our algorithms:
```
$ pip install -r requirements.txt
```

(Optional - extra info)
If you ever want to switch environments, first deactivate your current env, then activate the desired one:
```
deactivate
source <name_of_other_env>/bin/activate
```

### VSCode Jupyter Notebook Extension
In the example I have included some jupyter notebooks as they are more interactive. You can view and edit them with vs code and the jupyter notebook extension. You can alternatively use [jupyter notebooks natively](https://jupyter.org/), but I prefer and recommend to use VSCode and the extension.

## <a name="fundamental-prep"></a>Fundamental Prep

### Python
Most of the code we will be writing will be in python. We will have an education session that covers python, but we have provided some supplemental guides and study material below:
- [Python Learn by example](https://python-by-examples.readthedocs.io/en/latest/)
- [Python Tutorials](https://www.learnpython.org/en/Hello%2C_World%21)

There are many more resources online and if you're ever confused make sure to use resources like ChatGPT, stackoverflow, and others to your advantage!

### <a name="git"></a>Git
We also recommend using git with your team to manage your code

Git is a version control system for software. It allows developers to work on code simultaneously and then merge there code together. There are many other things you can do with git as well. If you want more info you can go here to see some git [tutorials](https://www.w3schools.com/git/git_intro.asp?remote=github)

## <a name="machine-learning"></a>Machine Learning & Local Training

Machine learning can be a powerful tool for building trading algorithms — from predicting price movements to classifying market regimes. This section covers how to train models locally on your machine.

### Key Libraries

The `requirements.txt` includes the core ML libraries you'll need:
- **scikit-learn** — classical ML models (linear regression, random forests, SVMs, etc.)
- **pandas** / **numpy** — data manipulation and feature engineering
- **matplotlib** — visualizing training results and model performance

### General Workflow

A typical ML-based strategy follows this pattern:

1. **Prepare your data** — load historical price/volume data, engineer features (e.g. moving averages, RSI, returns)
2. **Split your data** — use an earlier window for training and a later window for testing (never shuffle time-series data randomly)
3. **Train your model** — fit on the training set
4. **Evaluate** — measure performance on the held-out test set using metrics like accuracy, Sharpe ratio, or custom PnL
5. **Iterate** — tune hyperparameters, add features, and re-evaluate

### Example: Training a Simple Model Locally

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your data
df = pd.read_csv("market_data.csv")

# Feature engineering
df["returns"] = df["close"].pct_change()
df["ma_5"] = df["close"].rolling(5).mean()
df["target"] = (df["returns"].shift(-1) > 0).astype(int)  # 1 if next day is up
df = df.dropna()

# Train/test split (time-based)
split = int(len(df) * 0.8)
X = df[["returns", "ma_5"]]
y = df["target"]

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
```

### Tips & Pitfalls

- **Avoid lookahead bias** — never use future data as a feature when training
- **Be wary of overfitting** — a model that performs perfectly on training data but fails on test data is useless in live trading
- **Keep it simple first** — a linear model with good features often beats a complex model with bad ones
- **Use cross-validation carefully** — for time-series, use [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) instead of standard k-fold

### Resources
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Hands-On Machine Learning with Scikit-Learn (free preview)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Advances in Financial Machine Learning (Lopez de Prado)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — highly recommended for quant ML

---

Now you can develop your algorithm! Happy coding!
