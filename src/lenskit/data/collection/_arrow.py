from typing import Generic, Iterator, overload

import pandas as pd
import pyarrow as pa

from ..items import ItemList
from ._base import ID, KL, ItemListCollection
from ._keys import create_key_type


class ArrowILC(Generic[KL], ItemListCollection[KL]):
    """
    Item list collection packed into a single Arrow table.
    """

    _table: pa.Table
    _keys: pa.Table
    _lists: pa.ChunkedArray
    _list_cache: list[None | ItemList] | None = None
    _index: pd.Index | pd.MultiIndex

    def __init__(self, table: pa.Table):
        super().__init__([n for n in table.column_names if n != "items"])
        self._table = table
        self._init()

    def _init(self):
        key = [n for n in self._table.column_names if n != "items"]
        self._key_class = create_key_type(*key)
        self._keys = self._table.drop_columns("items")
        self._lists = self._table.column("items")
        self._index = pd.MultiIndex.from_arrays(
            [c.to_numpy() for c in self._keys.columns], names=self._keys.column_names
        )

    def record_batches(self, batch_size=5000, columns=None):
        return self._table.to_batches()

    @property
    def list_schema(self) -> dict[str, pa.DataType]:
        """
        Get the schema for the lists in this ILC.
        """
        field = self._table.field("items")
        lt = field.type.value_type
        return {f.name: f.type for f in lt.fields}

    def items(self) -> Iterator[tuple[KL, ItemList]]:
        "Iterate over item lists and keys."
        for i in range(self._table.num_rows):
            for i, k in enumerate(self._keys.to_pylist()):
                key = self.key_type(**k)
                yield key, self._get_list(i)

    @overload
    def lookup(self, key: tuple) -> ItemList | None: ...
    @overload
    def lookup(self, *key: ID, **kwkey: ID) -> ItemList | None: ...
    def lookup(self, *args, **kwargs) -> ItemList | None:
        if len(args) != 1 or not isinstance(args[0], tuple):
            key = self._key_class(*args, **kwargs)
        else:
            key = args[0]

        try:
            pos = self._index.get_loc(tuple(key))
        except KeyError:
            return None

        assert isinstance(pos, int)
        return self[pos][1]

    def _get_list(self, i) -> ItemList:
        if self._list_cache is None:
            self._list_cache = [None for _j in range(len(self))]
        cached = self._list_cache[i]

        if cached is None:
            il_data = self._lists[i].values
            cached = ItemList.from_arrow(il_data)
            self._list_cache[i] = cached
        return cached

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, pos: int, /) -> tuple[KL, ItemList]:
        k = {n: self._keys.column(i)[pos] for (i, n) in enumerate(self._keys.column_names)}
        return self.key_type(**k), self._get_list(pos)

    def __getstate__(self):
        return {"table": self._table}

    def __setstate__(self, state):
        self._table = state["table"]
        self._init()
