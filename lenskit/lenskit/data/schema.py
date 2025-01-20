"""
Pydantic models for LensKit data schemas.  These models define define the data
schema in memory and also define how schemas are serialized to and from
configuration files.  See :ref:`data-model` for details.

.. note::

    The schema does not specify data types directly — data types are inferred
    from the underlying Arrow data structures.  This reduces duplication of type
    information and the opportunity for inconsistency.
"""

# pyright: strict
from __future__ import annotations

from enum import Enum
from typing import Literal, OrderedDict

from pydantic import BaseModel


class AttrLayout(Enum):
    SCALAR = "scalar"
    """
    Scalar (non-list, non-vector) attribute value.
    """
    LIST = "list"
    """
    Homogenous, variable-length list of attribute values.
    """
    VECTOR = "vector"
    """
    Homogenous, fixed-length vector of numeric attribute values.
    """
    SPARSE = "sparse"
    """
    Homogenous, fixed-length sparse vector of numeric attribute values.
    """


class DataSchema(BaseModel, extra="forbid"):
    """
    Description of the entities and layout of a dataset.
    """

    entities: dict[str, EntitySchema] = {}
    """
    Entity classes defined for this dataset.
    """
    relationships: dict[str, RelationshipSchema] = {}
    """
    Relationship classes defined for this dataset.
    """


class EntitySchema(BaseModel, extra="forbid"):
    """
    Entity class definitions in the dataset schema.
    """

    id_type: Literal["int", "str", "uuid"] | None = None
    """
    The data type for identifiers in this entity class.
    """
    attributes: dict[str, ColumnSpec] = {}
    """
    Entity attribute definitions.
    """


class RelationshipSchema(BaseModel, extra="forbid"):
    """
    Relationship class definitions in the dataset schema.
    """

    entities: OrderedDict[str, str | None]
    """
    Define the entity classes participating in the relationship.  For aliased
    entity classes (necessary for self-relationships), the key is the alias, and
    the value is the original entity class name.
    """
    interaction: bool = False
    """
    Whether this relationship class records interactions.
    """
    attributes: dict[str, ColumnSpec] = {}
    """
    Relationship attribute definitions.
    """


class ColumnSpec(BaseModel, extra="forbid"):
    layout: AttrLayout = AttrLayout.SCALAR
    """
    The attribute layout (whether and how multiple values are supported).
    """
