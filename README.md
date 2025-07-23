# Elastic Representation: Mitigating Spurious Correlations for Group Robustness

This is the official implementation of the [paper]() "Elastic Representation: Mitigating Spurious Correlations for Group Robustness" published at AISTATS 2025.

Authors: Tao Wen, Zihan Wang, Quan Zhang, and Qi Lei

## Abstract

Deep learning models can suffer from severe performance degradation when relying on spurious correlations between input features and labels, making the models perform well on training data but have poor prediction accuracy for minority groups. This problem arises especially when training data are limited or imbalanced. While most prior work focuses on learning invariant features (with consistent correlations to y), it overlooks the potential harm of spurious correlations between features. We hereby propose Elastic Representation (ElRep) to learn features by imposing Nuclear- and Frobenius-norm penalties on the representation from the last layer of a neural network. Similar to the elastic net, ElRep enjoys the benefits of learning important features without losing feature diversity.

## Prerequisite

This implementation is based on [PDE](https://github.com/uclaml/PDE) and [Wilds](https://github.com/p-lambda/wilds). We thank the authors for their great work. 

Our modification is mainly adding the regularization in the objective function:

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
```
@InProceedings{pmlr-v258-wen25a,
  title = 	 {Elastic Representation: Mitigating Spurious Correlations for Group Robustness},
  author =       {Wen, Tao and Wang, Zihan and Zhang, Quan and Lei, Qi},
  booktitle = 	 {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {541--549},
  year = 	 {2025},
  editor = 	 {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume = 	 {258},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--05 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v258/main/assets/wen25a/wen25a.pdf},
  url = 	 {https://proceedings.mlr.press/v258/wen25a.html},
  abstract = 	 {Deep learning models can suffer from severe performance degradation when relying on spurious correlations between input features and labels, making the models perform well on training data but have poor prediction accuracy for minority groups. This problem arises especially when training data are limited or imbalanced. While most prior work focuses on learning invariant features (with consistent correlations to y), it overlooks the potential harm of spurious correlations between features. We hereby propose Elastic Representation (ElRep) to learn features by imposing Nuclear- and Frobenius-norm penalties on the representation from the last layer of a neural network. Similar to the elastic net, ElRep enjoys the benefits of learning important features without losing feature diversity. The proposed method is simple yet effective. It can be integrated into many deep learning approaches to mitigate spurious correlations and improve group robustness. Moreover, we theoretically show that ElRep has minimum negative impacts on in-distribution predictions. This is a remarkable advantage over approaches that prioritize minority groups at the cost of overall performance.}
}
```