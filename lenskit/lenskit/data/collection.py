from __future__ import annotations

from types import UnionType
from typing import (
    Hashable,
    Literal,
    TypeAlias,
)

from .items import ItemList
from .types import EntityId

KeySchema: TypeAlias = dict[str, type[Hashable] | UnionType]
ListKey: TypeAlias = dict[str, Hashable]
InternalKey: TypeAlias = tuple[Hashable, ...] | Hashable

UserIDSchema: KeySchema = {"user_id": EntityId}
"""
Schema consisting just of a user ID.
"""


class ItemListCollection:
    """
    A collection of item lists, identified by keys.  It is like a dictionary,
    but knows about components of its composite keys, and can interface with
    data frame and on-disk storage formats.  Keys are defined by a *schema* of
    fields with associated types.

    An item list collection allows a superset of its key fields to be passed to
    the query method :meth:`get_list`; extra fields are ignored.

    Args:
        schema:
            The key schema, as a dictionary mapping fields to their types.  If
            unspecified, uses :data:`UserIDSchema`, consisting of a single
            ``user_id`` field.
    """

    _key_schema: KeySchema
    _data: dict[InternalKey, ItemList]

    def __init__(
        self,
        schema: KeySchema = UserIDSchema,
    ):
        """
        Construct a collection with a specified schema.
        """
        self._key_schema = schema
        self._data = {}

    @classmethod
    def from_dict(
        cls,
        data: dict[InternalKey | Hashable, ItemList],
        schema: KeySchema = UserIDSchema,
    ):
        pass

    def _make_key(self, fields: ListKey, *, extra: Literal["ignore", "error"]) -> InternalKey:
        x = tuple(fields[k] for k in self._key_schema.keys())
        if extra == "error" and len(x) < len(fields):
            efs = set(fields.keys()) - set(self._key_schema.keys())
            raise KeyError(f"extra keys {efs}")

        if len(x) == 1:
            return x[0]
        else:
            return x

    def get_list(self, **fields: Hashable) -> ItemList:
        """
        Get the list with the specified key.

        Args:
            fields:
                The fields of the key.  All fields defined by the key's schema
                must be passed; any fields that are not defined in the schema
                are ignored.

        Returns:
            The item list with the specified key.
        """
        return self._data[self._make_key(fields, extra="ignore")]

    def add_list(self, list: ItemList, **fields: Hashable) -> None:
        """
        Add the list with a specified key.

        Args:
            list:
                The item list to add.
            fields:
                The fields of the key.  Must be exactly the set defined in the schema.
        """

        self._data[self._make_key(fields, extra="error")] = list
