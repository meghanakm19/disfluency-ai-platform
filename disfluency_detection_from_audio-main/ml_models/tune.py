"""Simple hyperparameter tuning harness for baseline classifiers."""
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from ml_models.classical_baselines import build_dataset


def tune_and_save(manifest, out_dir):
    X, y = build_dataset(manifest)
    param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm = SVC(kernel='rbf', probability=True)
    grid = GridSearchCV(svm, param_grid_svm, cv=3, scoring='f1_macro')
    grid.fit(X, y)
    best = grid.best_estimator_
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out / 'svm_tuned.joblib')
    print('Best SVM params:', grid.best_params_)

    param_grid_rf = {'n_estimators': [50,100,200], 'max_depth': [None, 10, 20]}
    rf = RandomForestClassifier()
    grid2 = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1_macro')
    grid2.fit(X, y)
    joblib.dump(grid2.best_estimator_, out / 'rf_tuned.joblib')
    print('Best RF params:', grid2.best_params_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    tune_and_save(args.manifest, args.out)
