import pickle

import numpy as np
import pandas as pd

from pytest import raises

from lenskit.data import ItemList
from lenskit.data.collection import ItemListCollection, UserIDKey, _create_key, project_key


def test_generic_key():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    assert isinstance(usk, tuple)
    assert usk == ("alphabet", 42)
    assert usk.user_id == "alphabet"  # type: ignore
    assert usk.seq_no == 42  # type: ignore


def test_pickle_generic():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    bs = pickle.dumps(usk)
    usk2 = pickle.loads(bs)

    assert isinstance(usk2, tuple)
    assert usk2 == usk


def test_project_key():
    usk = _create_key(("user_id", "seq_no"), "alphabet", 42)

    uk = project_key(usk, UserIDKey)
    assert isinstance(uk, UserIDKey)
    assert uk.user_id == "alphabet"
    assert uk == ("alphabet",)


def test_project_missing_fails():
    usk = _create_key(("seq_no",), 42)

    with raises(TypeError, match="missing field"):
        project_key(usk, UserIDKey)


def test_collection_empty():
    empty = ItemListCollection(UserIDKey)
    assert len(empty) == 0
    with raises(StopIteration):
        next(iter(empty))


def test_collection_add_item():
    ilc = ItemListCollection(UserIDKey)

    ilc.add(ItemList(["a"]), 72)
    assert len(ilc) == 1
    key, il = next(iter(ilc))
    assert key.user_id == 72
    assert np.all(il.ids() == ["a"])

    il2 = ilc.lookup(72)
    assert il2 is il


def test_collection_from_dict_nt_key():
    ilc = ItemListCollection.from_dict({UserIDKey(72): ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_tuple_key():
    ilc = ItemListCollection.from_dict({(72,): ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_singleton_key():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key=UserIDKey)

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_collection_from_dict_singleton_field():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key="user_id")

    assert len(ilc) == 1
    k, il = ilc[0]
    assert k.user_id == 72  # type: ignore
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_lookup_projected():
    ilc = ItemListCollection.from_dict({72: ItemList(["a"])}, key="user_id")

    usk = _create_key(("user_id", "seq"), 72, 100)
    il = ilc.lookup_projected(usk)
    assert il is not None
    assert len(il) == 1
    assert np.all(il.ids() == ["a"])


def test_add_from():
    ilc = ItemListCollection(["model", "user_id"])

    ilc1 = ItemListCollection.from_dict({72: ItemList(["a", "b"])}, key="user_id")
    ilc.add_from(ilc1, model="foo")

    assert len(ilc) == 1
    il = ilc.lookup(("foo", 72))
    assert il is not None
    assert il.ids().tolist() == ["a", "b"]


def test_from_df(rng, ml_ratings: pd.DataFrame):
    ml_ratings = ml_ratings.rename(columns={"user": "user_id", "item": "item_id"})
    ilc = ItemListCollection.from_df(ml_ratings, UserIDKey)
    assert len(ilc) == ml_ratings["user_id"].nunique()
    assert set(k.user_id for k in ilc.keys()) == set(ml_ratings["user_id"])

    for uid in rng.choice(ml_ratings["user_id"].unique(), 25):
        items = ilc.lookup(user_id=uid)
        udf = ml_ratings[ml_ratings["user_id"] == uid]
        assert len(items) == len(udf)
        assert np.all(np.unique(items.ids()) == np.unique(udf["item_id"]))

    tot = sum(len(il) for il in ilc.lists())
    assert tot == len(ml_ratings)


def test_to_df():
    ilc = ItemListCollection.from_dict(
        {72: ItemList(["a"], scores=[1]), 82: ItemList(["a", "b", "c"], scores=[3, 4, 10])},
        key="user_id",
    )
    df = ilc.to_df()
    print(df)
    assert len(df) == 4
    assert df["user_id"].tolist() == [72, 82, 82, 82]
