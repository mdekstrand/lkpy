import numpy as np
import torch

import hypothesis.extra.numpy as nph
import hypothesis.strategies as st
from hypothesis import given
from pytest import approx

from lenskit.util.heap import BatchedMinHeap


def test_empty_heap():
    heap = BatchedMinHeap(10, 5, None)
    assert torch.all(heap.sizes == 0)
    assert torch.all(heap.values == 0.0)
    assert heap.extra is None


def test_add_one():
    heap = BatchedMinHeap(1, 5, None)
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([3.5], dtype=torch.float32), None
    )
    assert heap.sizes[0] == 1
    assert heap.values[0, 0] == 3.5
    vs = heap.row_values(0)
    assert len(vs) == 1
    assert vs[0] == 3.5


def test_add_drop():
    heap = BatchedMinHeap(1, 2, None)
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([3.5], dtype=torch.float32), None
    )
    assert heap.sizes[0] == 1
    assert heap.values[0, 0] == 3.5
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([4.5], dtype=torch.float32), None
    )
    assert heap.sizes[0] == 2
    assert heap.values[0, 0] == 3.5
    assert heap.values[0, 1] == 4.5
    # this value is smaller, should not be isnerted
    heap.insert(torch.tensor([0], dtype=torch.int32), torch.tensor([2], dtype=torch.float32), None)

    assert heap.sizes[0] == 2
    assert heap.values[0, 0] == 3.5
    assert heap.values[0, 1] == 4.5
    vs = heap.row_values(0)
    assert len(vs) == 2
    assert vs.numpy() == approx([3.5, 4.5])


def test_add_replace():
    heap = BatchedMinHeap(1, 2, None)
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([3.5], dtype=torch.float32), None
    )
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([4.5], dtype=torch.float32), None
    )
    # this value is smaller, should not be isnerted
    heap.insert(
        torch.tensor([0], dtype=torch.int32), torch.tensor([5.5], dtype=torch.float32), None
    )

    assert heap.sizes[0] == 2
    assert heap.values[0, 0] == 4.5
    assert heap.values[0, 1] == 5.5
    vs = heap.row_values(0)
    assert len(vs) == 2
    assert vs == approx([4.5, 5.5])


def test_add_replace_several():
    heap = BatchedMinHeap(1, 5, None)
    for i in range(5):
        heap.insert(
            torch.tensor([0], dtype=torch.int32), torch.tensor([i], dtype=torch.float32), None
        )

    heap.insert(torch.tensor([0], dtype=torch.int32), torch.tensor([10], dtype=torch.float32), None)
    assert heap.sizes[0] == 5
    assert torch.any(heap.values[0, :] == 10)
    assert torch.all(heap.values[0, :] > 0)

    # this value is smaller, should not be isnerted
    heap.insert(torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.float32), None)

    assert heap.sizes[0] == 5
    assert torch.any(heap.values[0, :] == 10)
    assert torch.all(heap.values[0, :] > 0)


def test_add_replace_weird():
    # reproduce a controlled version of an odd hypothesis failure
    size = 9
    leading_zeros = 6
    negatives = 3
    post_zeros = 4

    heap = BatchedMinHeap(1, size, None)
    for i in range(leading_zeros):
        heap.insert(
            torch.tensor([0], dtype=torch.int32), torch.tensor([0.0], dtype=torch.float32), None
        )
        assert heap.sizes[0] == min(i + 1, size)

    for i in range(negatives):
        heap.insert(
            torch.tensor([0], dtype=torch.int32), torch.tensor([-1.0], dtype=torch.float32), None
        )
        assert heap.sizes[0] == min(i + 1 + leading_zeros, size)

    assert heap.sizes[0] == size
    assert heap.values[0, 0] == -1.0
    assert torch.sum(heap.values[0, :]) == -negatives

    # start overwriting -1s
    for i in range(post_zeros):
        print("overwrite", i, "initial sum", torch.sum(heap.values[0, :]))
        heap.insert(
            torch.tensor([0], dtype=torch.int32), torch.tensor([0.0], dtype=torch.float32), None
        )
        print("overwrite", i, "post sum", torch.sum(heap.values[0, :]))
        assert heap.sizes[0] == size
        assert torch.sum(heap.values[0, :]) == -max(negatives - i - 1, 0)


@given(
    st.integers(min_value=1, max_value=100),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), max_size=1000),
)
def test_single_heap(size: int, vals: list[float]):
    heap = BatchedMinHeap(1, size, None)

    for i, v in enumerate(vals, 1):
        heap.insert(torch.tensor([0], dtype=torch.int32), torch.tensor([v]), None)
        assert heap.row_size(0) == min(i, size)
        assert heap.values[0, 0] <= torch.min(heap.values[0, : min(i, size)])

        sorted = np.sort(vals[:i])

        try:
            if i <= size:
                assert np.all(sorted == np.sort(heap.row_values(0)))
            else:
                assert sorted[-size:] == approx(np.sort(heap.row_values(0)))

        except AssertionError as e:
            print("size", size)
            print("iter", i)
            print("values", np.array(vals))
            print("heap", heap.values[0])
            raise e


@given(
    st.data(),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=100),
)
def test_many_heaps(data: st.DataObject, batches: int, size: int, rounds: int):
    heap = BatchedMinHeap(batches, size, None)
    drawn = [[] for i in range(batches)]

    for i in range(rounds):
        rows = data.draw(st.sets(st.integers(0, batches - 1), min_size=1, max_size=batches))
        rows = np.array(list(rows), np.int32)
        vals = data.draw(
            nph.arrays(
                np.float32,
                len(rows),
                elements=st.floats(allow_infinity=False, allow_nan=False, width=32),
            )
        )

        for r, v in zip(rows, vals):
            drawn[r].append(v)

        rows = torch.from_numpy(rows)
        vals = torch.from_numpy(vals)
        sizes = heap.sizes[rows]

        heap.insert(rows, vals, None)

        assert torch.all(heap.sizes[rows] >= sizes)

    for i in range(batches):
        sz = heap.sizes[i].item()
        assert sz == min(len(drawn[i]), size)
        sorted = np.sort(drawn[i])
        if sz < size:
            assert np.sort(heap.values[i, :sz]) == approx(sorted)
        else:
            assert np.sort(heap.values[i, :]) == approx(sorted[-sz:])
