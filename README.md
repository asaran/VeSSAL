# VolumE Sampling for Streaming Active Learning (VeSSAL)

This repository is an implementation of the VeSSAL batch active learning algorithm. Details are in the following [paper](https://arxiv.org/abs/2303.02535):
> A. Saran, S. Yousefi, A. Krishnamurthy, J. Langford, J.T. Ash.
Streaming Active Learning with Deep Neural Networks. ICML 2023.

For a quick overview of the approach, please check the [talk](https://icml.cc/virtual/2023/poster/24291) and [poster](https://icml.cc/media/PosterPDFs/ICML%202023/24291.png?t=1690414066.469752) presented at ICML 2023.

This code was built on [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning), and [Batch Active learning by Diverse Gradient Embeddings](https://github.com/JordanAsh/badge).

# Dependencies

To run this code fully, you'll need [PyTorch and Torchvision](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/stable/). We've tested our code with PyTorch 1.8.0, 1.13.0 and Python 3.7, 3.8.

# Running an experiment

`python run.py --model resnet --nQuery 1000 --data CIFAR10 --alg vessal` \
runs an active learning experiment using a ResNet and CIFAR-10 data, querying batches of 1,000 samples according to the VeSSAL algorithm.
This code allows you to also run each of the baseline algorithms used in our paper. 

`python run.py --model mlp --nQuery 10000 --data SVHN --alg conf`\
runs an active learning experiment using an MLP and SVHN data, querying batches of 10,000 with confidence sampling.

# Bibliography
If you find our work to be useful in your research, please cite:
```
@article{saran2023streaming,
  title={Streaming Active Learning with Deep Neural Networks},
  author={Saran, Akanksha and Yousefi, Safoora and Krishnamurthy, Akshay
  and Langford, John and Ash, Jordan T.},
  journal={arXiv preprint arXiv:2303.02535},
  year={2023}
}
```
