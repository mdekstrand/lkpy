"""
TorchScript-compatible bulk heap support.
"""

import torch


@torch.jit.script
def _batch_swap(data: torch.Tensor, rows: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor):
    saved = data[rows, i1]
    data[rows, i1] = data[rows, i2]
    data[rows, i2] = saved


@torch.jit.script
class BatchedMinHeap:
    """
    A batch of min-heaps that can store an additional data item.  This allows
    efficient management of multiple min-heaps, callable from TorchScript, for
    things like incremental batchwise top-K operations.

    Values must be float32; extra data, if used, can be of an arbitrary PyTorch dtype.
    """

    size: int
    values: torch.Tensor
    extra: torch.Tensor
    sizes: torch.Tensor

    def __init__(self, heap_count: int, heap_size: int, extra: torch.dtype = torch.float32):
        self.size = heap_size
        self.values = torch.zeros((heap_count, heap_size), dtype=torch.float32)
        self.sizes = torch.zeros((heap_count,), dtype=torch.int32)
        self.extra = torch.zeros_like(self.values, dtype=extra)

    def insert(
        self,
        rows: torch.Tensor,
        values: torch.Tensor,
        extra: torch.Tensor | None,
    ):
        """
        Insert values into the heap at the specified rows.  Returns the values
        removed to make place for these values.
        """
        rsz = self.sizes[rows]
        # to insert: rows that are not full
        m_insert = rsz < self.size
        # to replace: rows thare are full, and the new value is greater than the smallest
        m_replace = (rsz == self.size) & (self.values[rows, 0] < values)

        self._insert_new(
            rows[m_insert], values[m_insert], None if extra is None else extra[m_insert]
        )
        self._insert_replace(
            rows[m_replace], values[m_replace], None if extra is None else extra[m_replace]
        )

        m_add = m_insert | m_replace
        assert torch.all(self.values[rows[m_add], 0] <= values[m_add])
        assert torch.all(values[~m_add] <= self.values[rows[~m_add], 0])

    def row_values(self, row: int) -> torch.Tensor:
        """
        Get the values for a single row in this heap.
        """
        limit = self.sizes[0]
        return self.values[row, :limit]

    def row_extra(self, row: int) -> torch.Tensor | None:
        """
        Get the extra values for a single row in this heap.
        """
        if self.extra is not None:
            limit = self.sizes[0]
            return self.extra[row, :limit]
        else:
            return None

    def row_size(self, row: int) -> int:
        return self.sizes[row].item()  # type: ignore

    def _insert_new(
        self,
        rows: torch.Tensor,
        values: torch.Tensor,
        extra: torch.Tensor | None,
    ):
        rsz = self.sizes[rows]
        self.values[rows, rsz] = values
        if extra is not None:
            self.extra[rows, rsz] = extra
        self._upheap(rows)
        self.sizes[rows] += 1

    def _insert_replace(
        self,
        rows: torch.Tensor,
        values: torch.Tensor,
        extra: torch.Tensor | None,
    ):
        self.values[rows, 0] = values
        if extra is not None:
            self.extra[rows, 0] = extra
        self._downheap(rows)

    def _upheap(self, rows: torch.Tensor):
        pos = self.sizes[rows].clone()
        mask = pos > 0
        while torch.any(mask):
            i_row = rows[mask]
            i_pos = pos[mask]
            i_parent = (i_pos - 1) // 2
            # find the rows where the parent is greater than the child
            m_swap = self.values[i_row, i_parent] > self.values[i_row, i_pos]
            _batch_swap(self.values, i_row[m_swap], i_parent[m_swap], i_pos[m_swap])
            _batch_swap(self.extra, i_row[m_swap], i_parent[m_swap], i_pos[m_swap])
            pos = i_parent
            rows = i_row
            pos[~m_swap] = 0
            mask = pos > 0

    def _downheap(self, rows: torch.Tensor):
        finished = torch.zeros_like(rows, dtype=torch.bool)
        current = torch.zeros_like(rows, dtype=torch.int32)
        while not torch.all(finished):
            mins = current.clone()
            left = 2 * current + 1
            right = 2 * current + 2

            m_left = left < self.size
            ml = m_left.clone()
            m_left[ml] = self.values[rows[ml], left[ml]] < self.values[rows[ml], mins[ml]]
            mins[m_left] = left[m_left]

            m_right = right < self.size
            mr = m_right.clone()
            m_right[mr] = self.values[rows[mr], right[mr]] < self.values[rows[mr], mins[mr]]
            mins[m_right] = right[m_right]

            m_swap = mins != current
            _batch_swap(self.values, rows[m_swap], mins[m_swap], current[m_swap])
            _batch_swap(self.extra, rows[m_swap], mins[m_swap], current[m_swap])
            current = mins
            finished[~m_swap] = True
