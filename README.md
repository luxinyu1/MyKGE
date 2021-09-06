# MyKGE
This repo is the main task of my research internship at Zhejiang University AZFT Joint Lab for Knowledge Engine. It contains the reproduction of several representative KG Embedding methods.
I'm still working on adding more embedding models and improving the performance of current models, any helpful issues and pull requests are welcomed.

## Requirements

- torch**==**1.9.0+cu111
- tensorboard**==**2.6.0

## Usage
Run scripts under ```./scripts/``` directory. For example, you can use the following command to train a RotatE model and evaluate its performance:

```shell
./scripts/RotatE_train.sh
```

When ```--use-tensorboard``` is activated during training, you can start tensorboard on ```./logs/tensorboard/``` to visualize the computational graphs, like:

```
tensorboard --logdir ./logs/tensorboard/
```

## Supported Models

- RotatE
- TransE
- BoxE

I'm planning to add KG-BERT, ConvE in the future.
