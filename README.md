# ElRep
The code for Elastic Representation: Mitigating Spurious Correlations for Group Robustness @ AISTATS'25.

## Prerequisite

[torch torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric](https://pytorch.org/get-started/locally/)

[Wilds](https://github.com/p-lambda/wilds)

[Transformers](https://huggingface.co/docs/transformers/en/installation)

## Example commands

Note **theta1** and **theta2** stand for the weights of NN and FN, respectively. The commands below are for ERM + ElRep. You can replace ERM with the algorithm of your choice.

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
