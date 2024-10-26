"""
Item list serialization tests.
"""

import pickle

import pyarrow as pa

from pytest import fixture, mark

from lenskit.splitting import LastN, sample_users
from lenskit.util.test import ml_20m


@fixture(scope="session")
def test_users(ml_20m):
    split = sample_users(ml_20m, 5000, LastN(10))

    yield split.test


@mark.benchmark(max_time=10)
def test_il_pickle(test_users, benchmark):
    def serialize():
        for items in test_users.values():
            bs = pickle.dumps(items, pickle.HIGHEST_PROTOCOL)
            assert len(bs) > 0

    benchmark(serialize)


@mark.benchmark(max_time=10)
def test_il_pickle_df(test_users, benchmark):
    def serialize():
        for items in test_users.values():
            df = items.to_df()
            bs = pickle.dumps(df, pickle.HIGHEST_PROTOCOL)
            assert len(bs) > 0

    benchmark(serialize)


@mark.benchmark(max_time=10)
def test_il_df_tbl(test_users, benchmark):
    def serialize():
        for items in test_users.values():
            df = items.to_df()
            tbl = pa.Table.from_pandas(df)
            stream = pa.BufferOutputStream()
            write = pa.ipc.new_stream(stream, tbl.schema)
            write.write_table(tbl)
            write.close()
            buf = stream.getvalue()
            assert len(buf) > 0

    benchmark(serialize)
