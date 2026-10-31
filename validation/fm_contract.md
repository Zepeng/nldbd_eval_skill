# Foundation-model contract (fine-tuning input)

A foundation model is one Python module (file path or import name) with:

```python
EMBED_DIM: int | dict[str, int]                 # D, the encoder output width (one value, or one per dataset)
MODEL_INFO: dict                                # free-form description, recorded in every run

def to_tensors(dataset: str, inputs) -> tuple[torch.Tensor, ...]:
    """Turn what the reader yields for `dataset` (see the dataset cards) into the encoder's input
    tensors on CPU: your tokenizer / stem. Must not use labels, energies or categories."""

def load_encoder(dataset: str, pretrained: bool = True) -> torch.nn.Module:
    """The encoder for `dataset` (shared weights across datasets are your choice). With
    pretrained=True load your pretrained weights; with pretrained=False build the same
    architecture with random weights (the scratch control). forward(*tensors) returns either
    tokens [B, N, D] and a boolean mask [B, N] (True = valid), or an embedding [B, D]."""
```

Pretraining is yours. Record every event whose inputs it used (any split except test) and pass
that list to the fine-tuner with `--pretraining-ids pretraining_ids.npz` (array `event_id`);
it is merged into each run's `training_ids.npz` so gate G4a can check it.
`examples/tiny_fm.py` is a complete example.
