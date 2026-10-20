# TODO & Known Issues

## Remaining Tasks
- [ ] Extend training data to 5 years (2003-2007) for better generalization.
- [ ] Add static topographic features (slope, aspect) to high-res nodes.
- [ ] Implement the full QMSE loss exactly as described in the paper.
- [ ] Optimize training for GPU (CUDA).
- [ ] Write unit tests for data pipeline scripts.

## Known Issues
- Model correlation on extreme events (p99) is close to zero.
- The model still underperforms a simple "Spatial Mean" baseline.
- Training on CPU is extremely slow (~12 mins/epoch).