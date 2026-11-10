from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from inference.main import create_app
from tests import fakes
from training import train
from training.make_dataset import make_dataset
from training.provenance import dataset_provenance, read_git_commit


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "train.csv"
    make_dataset(path, rows=300)
    return path


def test_validate_dataset_accepts_good_data(dataset):
    assert train.validate_dataset(dataset)["rows"] == 300


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.head(10), "only 10 rows"),
    (lambda d: d.drop(columns=["feature_0"]), "missing columns"),
    (lambda d: d.assign(target=1), "single class"),
    (lambda d: d.assign(feature_1=None), "null"),
])
def test_validate_dataset_rejects_bad_data(tmp_path, dataset, mutate, msg):
    bad = tmp_path / "bad.csv"
    mutate(pd.read_csv(dataset)).to_csv(bad, index=False)
    with pytest.raises(train.DatasetValidationError, match=msg):
        train.validate_dataset(bad)


def test_preprocess_train_and_serve(tmp_path, dataset):
    split = train.preprocess(dataset, tmp_path / "out")
    assert split["train_rows"] + split["test_rows"] == 300
    model_path = tmp_path / "model.pkl"
    train.train_model(split["train_path"], model_path, n_estimators=10, max_depth=3)

    client = TestClient(create_app(str(model_path)))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["ready"] is True
    ok = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5]})
    assert ok.status_code == 200 and ok.json()["prediction"] in (0, 1)
    assert client.post("/predict", json={"features": [1.0]}).status_code == 422


def test_inference_not_ready_without_model(tmp_path):
    client = TestClient(create_app(str(tmp_path / "missing.pkl")))
    assert client.get("/health").status_code == 200  # alive...
    assert client.get("/ready").status_code == 503  # ...but must not receive traffic
    assert client.post("/predict", json={"features": [1.0] * 5}).status_code == 503


def test_provenance_reads_git_and_dvc_without_binaries(tmp_path):
    fakes.make_repo(tmp_path)
    prov = dataset_provenance(tmp_path)
    assert prov["dvc_md5"] == fakes.MD5 and prov["git_commit"] == fakes.COMMIT


def test_git_commit_from_packed_refs(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "packed-refs").write_text(f"# pack\n{fakes.COMMIT} refs/heads/main\n")
    assert read_git_commit(tmp_path) == fakes.COMMIT
