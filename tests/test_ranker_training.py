"""Ranker data regression tests with synthetic artifacts; no training or services."""

import polars as pl
import pytest
import torch

from src.training import train_ranker


@pytest.fixture
def artifacts(monkeypatch):
    frames = {
        "train.parquet": pl.DataFrame(
            {"user_id": [10, 10], "movie_id": [100, 101], "timestamp": [1, 2]},
            schema={"user_id": pl.Int32, "movie_id": pl.Int32, "timestamp": pl.Int64},
        ),
        "user_features.parquet": pl.DataFrame(
            {"user_id": [10], "user_avg_rating": [3.0]}
        ),
        "movie_features.parquet": pl.DataFrame(
            {"movie_id": [100, 101, 200, 300], "movie_avg_rating": [4.0] * 4}
        ),
    }
    # 300 has features but no embedding; 400 has an embedding but no features.
    mappings = {"user2idx": {20: 0, 10: 1},
                "movie2idx": {100: 0, 101: 1, 200: 2, 400: 3}}
    monkeypatch.setattr(
        train_ranker.pl, "read_parquet", lambda path: frames[path.rsplit("/", 1)[-1]]
    )

    def load(path, **kwargs):
        if path.endswith("mappings.pt"):
            return mappings
        return {}

    monkeypatch.setattr(train_ranker.torch, "load", load)

    class FakeModel:
        def __init__(self, num_users, num_movies):
            assert (num_users, num_movies) == (2, 4)

        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

        def user_tower(self, indices):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])[indices]

        def item_tower(self, indices):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]])[indices]

    monkeypatch.setattr(train_ranker, "TwoTowerModel", FakeModel)
    return frames, mappings


@pytest.mark.parametrize("sample_ratio", [0.5, 1.0])
def test_negatives_use_only_unseen_movies_with_features_and_embeddings(artifacts, sample_ratio):
    df = train_ranker.generate_training_data(sample_ratio=sample_ratio)
    negatives = df.filter(pl.col("target") == 0)
    positives = df.filter(pl.col("target") == 1)
    assert len(negatives) == len(positives) > 0
    assert negatives["movie_id"].to_list() == [200] * len(negatives)
    assert negatives["two_tower_score"].to_list() == pytest.approx([0.8] * len(negatives))
    expected_scores = {100: 0.0, 101: 1.0}
    for row in positives.iter_rows(named=True):
        assert row["two_tower_score"] == pytest.approx(expected_scores[row["movie_id"]])
    assert negatives["timestamp"].to_list() == positives["timestamp"].to_list()
    assert df["timestamp"].is_sorted()


@pytest.mark.parametrize("catalog", [[100, 101, 300], [300]])
def test_no_eligible_unseen_negative_fails_with_diagnostic(artifacts, catalog):
    frames, _ = artifacts
    frames["movie_features.parquet"] = frames["movie_features.parquet"].filter(
        pl.col("movie_id").is_in(catalog)
    )
    with pytest.raises(ValueError, match="No eligible unseen negative movies for user_id=10"):
        train_ranker.generate_training_data(sample_ratio=1.0)


def test_grouped_negative_sampling_preserves_user_timestamps(artifacts):
    frames, _ = artifacts
    frames["train.parquet"] = pl.DataFrame(
        {"user_id": [10, 20, 10], "movie_id": [100, 100, 101], "timestamp": [1, 2, 3]},
        schema={"user_id": pl.Int32, "movie_id": pl.Int32, "timestamp": pl.Int64},
    )
    frames["user_features.parquet"] = pl.DataFrame(
        {"user_id": [10, 20], "user_avg_rating": [3.0, 4.0]}
    )
    df = train_ranker.generate_training_data(sample_ratio=1.0)
    negatives = df.filter(pl.col("target") == 0)
    assert sorted(negatives.select("user_id", "timestamp").rows()) == [(10, 1), (10, 3), (20, 2)]
    seen = {10: {100, 101}, 20: {100}}
    expected_scores = {10: {200: 0.8}, 20: {101: 0.0, 200: 0.6}}
    for row in negatives.iter_rows(named=True):
        assert row["movie_id"] not in seen[row["user_id"]]
        assert row["two_tower_score"] == pytest.approx(
            expected_scores[row["user_id"]][row["movie_id"]]
        )


@pytest.mark.parametrize("mapping_name, missing_id, column", [
    ("user2idx", 10, "user_id"),
    ("movie2idx", 101, "movie_id"),
])
def test_missing_positive_ids_fail_instead_of_using_index_zero(artifacts, mapping_name, missing_id, column):
    _, mappings = artifacts
    del mappings[mapping_name][missing_id]
    with pytest.raises(ValueError, match=rf"Training {column} IDs missing.*{missing_id}"):
        train_ranker.generate_training_data(sample_ratio=1.0)
