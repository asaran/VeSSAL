# VolumE Sampling for Streaming Active Learning (VeSSAL)

This respository is an implementation of the VeSSAL batch active learning algorithm. Details are in the following [paper](https://asaran.github.io/papers/VeSSAL.pdf):
> A. Saran, S. Yousefi, A. Krishnamurthy, J. Langford, J.T. Ash.
Streaming Active Learning with Deep Neural Networks.

This code was built on [Kuan-Hao Huang's deep active learning repository](https://github.com/ej0cl6/deep-active-learning), and [Batch Active learning by Diverse Gradient Embeddings](https://github.com/JordanAsh/badge).

# Dependencies

To run this code fully, you'll need [PyTorch and Torchvision](https://pytorch.org/) (we're using version 1.8.0) and [scikit-learn](https://scikit-learn.org/stable/). We've been running our code in Python 3.7.

# Running an experiment

`python run.py --model resnet --nQuery 1000 --data CIFAR10 --alg vessal` \
runs an active learning experiment using a ResNet and CIFAR-10 data, querying batches of 1,000 samples according to the VeSSAL algorithm.
This code allows you to also run each of the baseline algorithms used in our paper. 

`python run.py --model mlp --nQuery 10000 --data SVHN --alg conf`\
runs an active learning experiment using an MLP and SVHN data, querying batches of 10,000 with confidence sampling.