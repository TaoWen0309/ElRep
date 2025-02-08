# ElRep
The code for Elastic Representation: Mitigating Spurious Correlations for Group Robustness @ AISTATS'25.

## Prerequisite

[torch torchvision torchaudio](https://pytorch.org/get-started/locally/)

and other related packages including:

torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

[Wilds](https://github.com/p-lambda/wilds)

[Transformers](https://huggingface.co/docs/transformers/en/installation)

## Example commands for ERM with Elastic Representation (ERM+ElRep)

Note **theta1** and **theta2** stand for the weights of NN and FN, respectively.

### Waterbirds

```
python run_expt.py --dataset waterbirds --algorithm ERM --seed 0 --theta1 0.001 --theta2 0.0001 --root_dir $path_to_your_data --log_dir ./logs --device 0 --download
```

### CelebA
```
python run_expt.py --dataset celebA --algorithm ERM --seed 0 --theta1 0.001 --theta2 0.0001 --root_dir $path_to_your_data --log_dir ./logs --device 0 --download
```

### CivilComments
```
python run_expt.py --dataset civilcomments --algorithm ERM --seed 0 --theta1 0.001 --theta2 0.0001 --root_dir $path_to_your_data --log_dir ./logs --device 0 --download
```
