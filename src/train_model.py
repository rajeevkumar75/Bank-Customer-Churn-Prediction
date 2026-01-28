# src/train_model.py

from pycaret.classification import (
    setup,
    create_model,
    tune_model,
    optimize_threshold,
    finalize_model,
    save_model,
    pull
)


def setup_environment(df):
    """
    PyCaret setup
    """
    setup(
        data=df,
        target="churn",

        normalize=True,
        normalize_method="zscore",

        fix_imbalance=True,

        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,

        session_id=42,
        fold=10,

        verbose=False
    )



def train_logistic_regression():
    """
    Logistic Regression (Recall-focused)
    """
    lr = create_model("lr")

    tuned_lr = tune_model(
        lr,
        optimize="Recall",
        n_iter=50,
        choose_better=True
    )

    tuned_lr = optimize_threshold(
        tuned_lr,
        optimize="Recall"
    )

    results = pull()
    return tuned_lr, results


def train_lightgbm():
    lgbm = create_model(
        "lightgbm",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        verbose=False
    )

    tuned_lgbm = tune_model(
        lgbm,
        optimize="AUC",
        n_iter=15,         
        choose_better=True
    )

    results = pull()
    return tuned_lgbm, results



def run_training_pipeline(df):
    setup_environment(df)

    tuned_lr, lr_results = train_logistic_regression()
    tuned_lgbm, lgbm_results = train_lightgbm()

    final_lr = finalize_model(tuned_lr)
    final_lgbm = finalize_model(tuned_lgbm)

    save_model(final_lr, "models/logistic_regression_churn")
    save_model(final_lgbm, "lightgbm_churn")

    return {
        "logistic_regression": lr_results,
        "lightgbm": lgbm_results
    }
