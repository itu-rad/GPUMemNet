# Related work

## Horus Formula
We chose a number of MLP neural network with varying their depth and width and monitoring GPU memory need of those models while training, then used the Horus formula to estimate the GPU memory need of those models.
- The data from the experiment to compare the actual GPU need of a deep learning task and the estimation from Horus formula | [Horus results excel file](Horus/horus_formula_not_working.xlsx)

- The experiment showed huge mispredictions with horus | [figure in PDF format](Horus/horus_formula_not_working.pdf)

## Fake Tensor Library
For this experiment, we used TIMM library models. We added some lines of code to enable estimation with fake tensor for the TIMM models. Also, we monitored GPU memory need of the models while training. We observed that fake tensor usually underestimates and for some models, it overestimates with huge differences, also we looked into the distribution of the differences.

- [Read more about fake tensor experiment](FakeTensor/README.md)