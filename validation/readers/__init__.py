"""Official split readers for the four NLDBD benchmark datasets.

Every reader exposes the same contract (see ``common.Reader``):

* ``table(split)``: event-aligned arrays for one split in the dataset's canonical order:
  ``event_id`` (str), ``label`` (int8, 1 = signal-like), ``energy_keV`` (float64),
  ``group`` (str), plus locator fields.
* ``iter_inputs(split, batch_size, indices=None)``: model inputs with exactly the paper's
  preprocessing, never containing energy or labels.

Label and energy conventions are those of the paper's evaluation (positive = signal-like).
"""
from .common import DATASETS, Reader, get_reader, data_root

__all__ = ["DATASETS", "Reader", "get_reader", "data_root"]
