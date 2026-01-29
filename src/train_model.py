from pycaret.classification import (
    setup, create_model, tune_model, finalize_model, save_model, pull
)

def setup_environment(df):
    target_col = 'churn' if 'churn' in df.columns else 'Exited'
    
    # Identify which columns are categorical (after manual LabelEncoding)
    # Even if they are numbers now, telling PyCaret they are categorical helps CatBoost.
    cat_features = ['gender', 'country', 'active_member', 'credit_card', 'is_senior']
    existing_cats = [col for col in cat_features if col in df.columns]

    setup(
        data=df,
        target=target_col,
        
        #I scaled manually in preprocessing.py
        normalize=False, 
        
        # Categorical handling
        categorical_features=existing_cats,
        
        # Balance handling: here SMOTE used by default
        fix_imbalance=True, 
        
        #Performance & Reproducibility
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
        session_id=42,
        fold=5,
        verbose=False
    )

#tuning the CatBoost model specifically for F1-score optimization:-
def train_catboost_classifier():

    cb = create_model("catboost")
    
    tuned_cb = tune_model(
        cb,
        optimize='F1', 
        n_iter=30,
        choose_better=True, #Ensuring: don't keep a "tuned" model that's worse than base
        custom_grid={
            'iterations': [200, 400, 600],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 9],
            'scale_pos_weight': [1, 3, 4]
        }
    )
    
    # Pull the final metrics table
    results = pull()
    return tuned_cb, results

#Orchestrates the training flow:-
def run_training_pipeline(df):
    
    setup_environment(df)
    
    print("🛠️  Tuning CatBoost parameters (this may take a minute)...")
    tuned_cb, cb_results = train_catboost_classifier()
    
    # Finalize trains on the full dataset (train + test) for production
    final_cb = finalize_model(tuned_cb)
    
    # Save with a clear versioning path if needed
    save_model(final_cb, "models/catboost_classifier_model")
    
    return {"catboost": cb_results}