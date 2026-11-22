import time

import numpy as np
from sklearn.dummy import DummyClassifier
from torchvision import datasets

import common


def main():
    common.set_seed(common.SEED)

    train_ds = datasets.CIFAR100(root=common.DATA_DIR, train=True)
    test_ds = datasets.CIFAR100(root=common.DATA_DIR, train=False)

    X_train = train_ds.data.reshape(len(train_ds), -1)
    y_train = np.array(train_ds.targets)
    X_test = test_ds.data.reshape(len(test_ds), -1)
    y_test = np.array(test_ds.targets)

    # 2 strategies frequent: guesses the same class & stratified: guesses randomly (with seed for repeated same results)
    for strategy in ["most_frequent", "stratified"]:
        model = DummyClassifier(strategy=strategy, random_state=common.SEED)

        start = time.perf_counter() # to get training time
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test) # confidence scores
        guess_time = time.perf_counter() - start

        print(strategy, common.compute_metrics(y_test, y_pred, probs))
        common.save_results(f"dummy_{strategy}", y_test, y_pred, probs,
                            train_time, guess_time, n_params=0)

if __name__ == "__main__":
    main()
