import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data.datasets import TabularDataset


def test_tabular_dataset_from_arrays_and_sampling():
    data = load_breast_cancer()
    dataset = TabularDataset.from_arrays(data.data, data.target, list(data.feature_names))

    assert dataset.n_features == data.data.shape[1]
    samples = dataset.sample(n=5, seed=123)
    assert samples.shape == (5, dataset.n_features)
    assert np.all(np.isfinite(samples))

    x, y = dataset.get_row(10)
    assert x.shape == (dataset.n_features,)
    assert y in {0, 1}

    train_ds, val_ds, test_ds = dataset.split(seed=42)
    assert sum(len(split) for split in (train_ds, val_ds, test_ds)) == len(dataset)
    assert train_ds.feature_names == dataset.feature_names


def test_tabular_dataset_from_dataframe():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["label"] = data.target

    dataset = TabularDataset.from_dataframe(df, label_col="label")
    assert dataset.n_features == data.data.shape[1]
    x, y = dataset.get_row(0)
    assert x.shape == (dataset.n_features,)
    assert y in {0, 1}
