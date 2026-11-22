import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common

def main():
    common.set_seed(common.SEED)
    X_train, y_train, X_val, y_val, X_test, y_test = common.get_numpy_splits()

    # testing different C values and keep the one with best val acc
    best_acc = 0
    for C in [0.01, 0.1, 1.0]:
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=200, random_state=common.SEED),
            LogisticRegression(C=C, max_iter=1000),
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        val_acc = model.score(X_val, y_val)
        print(f"C={C}: val accuracy {val_acc:.4f}, trained in {train_time:.1f}s")
        if val_acc > best_acc:
            best_acc, best_model, best_C, best_time = val_acc, model, C, train_time

    print(f"Best C={best_C}")

    # test set used once with best model only
    start = time.perf_counter()
    probs = best_model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    guess_time = time.perf_counter() - start

    pca = best_model.named_steps["pca"]
    lr = best_model.named_steps["logisticregression"]
    n_params = pca.components_.size + lr.coef_.size + lr.intercept_.size

    print("test", common.compute_metrics(y_test, y_pred, probs))
    common.save_results("logistic_regression", y_test, y_pred, probs,
                        best_time, guess_time, n_params)


if __name__ == "__main__":
    main()