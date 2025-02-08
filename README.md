# Code for Elastic Representation: Mitigating Spurious Correlations for Group Robustness

## Prerequisite

This implementation is heavily based on [PDE](https://github.com/uclaml/PDE). Our modification is mainly adding the regularization in the training process:

```
_, s, _ = torch.linalg.svd(features)
NN = self.theta1 * torch.sum(torch.abs(s))/features.shape[0]
FN = self.theta2 * torch.sum(torch.square(s))/features.shape[0]
objective += (NN+FN)
```

Where NN and FN stands for Nuclear Norm and Frobenius Norm, respectively.

It requires the following packages:

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

## Citation
Please consider citing our work if you find it useful: