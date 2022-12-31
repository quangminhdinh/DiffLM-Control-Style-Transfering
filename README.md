# Sequence-to-Sequence Continuous Diffusion Language Models for Control Style Transfering

This repository is the official implementation of the models introduced in [Sequence-to-Sequence Continuous Diffusion Language Models for Control Style Transfering](https://github.com/quangminhdinh/SDPE/blob/main/Sequence_to_Sequence_Continuous_Diffusion_Language_Models_for_Control_Style_Transfering.pdf).

The implementation is based on the [BERT replication](https://github.com/EBGU/Diffusion-LM) of [Diffusion-LM Improves Controllable Text Generation](https://github.com/XiangLi1999/Diffusion-LM).

```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```

## Setup

All models were trained using an NVIDIA A6000 GPU with 45 GiB RAM, in 20 epochs, with 250-500 diffusion steps. Details can be found in the paper.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

A pre-trained model is required as the backbone model for all of the models introduced in the paper. Some suggested models:

- [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)
- [bert-mini](https://huggingface.co/prajjwal1/bert-mini)
- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

The downloaded pre-trained model should be put inside `base/`.

## Training

To train the model, run this command:

```train
python -m train.py
```

All checkpoints can be found in `domains/[DATASET NAME]/checkpoints/`.
