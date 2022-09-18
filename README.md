## DogeTickets

This repository contains code and data for NLPCC 2022 paper titled [Doge Tickets: Uncovering Domain-general Lnaguage Models via Playing Lottery Tickets.](https://arxiv.org/abs/2207.09638).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

<!-- Probably you will think this as another *"empty"* repo of a preprint paper 🥱.
Wait a minute! The authors are working day and night 💪, to make the code and models available.
We anticipate the code will be out * **in one week** *. -->

* 9/5/22: We released our paper. Check it out!
* 9/1/22: We released our code and data. Check it out!

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Case Study](#case-study)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

It is recognized that, when LMs are faced with multiple domains, a critical portion of parameters behave unexpectedly in a domain-specific manner while others behave in a domain-general one. Motivated by this phenomenon, we for the first time posit that domain-general parameters can underpin a domain-general LM that can be derived from the original LM. To uncover the domain-general LM, we propose to identify domain-general parameters by playing lottery tickets (dubbed doge tickets).

<img src="assets/motivation.png" width="270" alt="case" align=center/> <img src="assets/method.png" width="300" alt="case" align=center/>

## Getting Started

### Requirements

- PyTorch
- Numpy

### Training

**Training scripts**

We provide example training scripts for MuG with and without the structural adapter. For example, in `scripts/run_google_bert_train.sh`, we provide an example for training MuG without the adapter. We explain the arguments in following:
* `--mode`: Train or evaluate the model.
* `--pretrained_model_name_or_path`: Pre-trained checkpoints to start with.
* `--embed_learning_rate`: Learning rate for BERT backbones and adapters.
* `--learning_rate`: Learning rate for modules built upon BERT backbones.
* `--hidden_size`: Size of hidden states.
* `--sentiment_size`: Number of types of sentiments.
* `--tag_size`: Number of types of tags
* `--use_adapter`: Use the adapter or not.

### Evaluation

We also provide example training scripts, for example `scripts/run_google_bert_eval.sh`, where arguments share similar meaning as those in training ones.

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Yi (`yang.yi@bit.edu.cn`) or Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code or data in your work:

```bibtex
@inproceedings{yang2022doge,
   title={Doge Tickets: Uncovering Domain-general Lnaguage Models via Playing Lottery Tickets},
   author={Yang, Yi and Zhang, Chen and Wang, Benyou and Song, Dawei},
   booktitle={NLPCC},
   year={2022}
}
```
