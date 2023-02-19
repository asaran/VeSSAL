import numpy as np
from .strategy import Strategy
import torch
import copy
from sklearn.random_projection import GaussianRandomProjection
import os
import time

class StreamingSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(StreamingSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.skipped = []

        if self.args["data"] == 'CLOW' or self.args["data"] == 'clip':
            self.transformer = GaussianRandomProjection(n_components=2560)
        self.zeta = self.args["zeta"]

    # just in case values get too big, sometimes happens
    def inf_replace(self, mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat

    def streaming_sampler(self, samps, k, early_stop=False, streaming_method='det', \
                        cov_inv_scaling=100, embs="grad_embs"):
        inds = []
        skipped_inds = []
        if embs == "penultimate":
            samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        dim = samps.shape[-1]
        rank = samps.shape[-2]

        covariance = torch.zeros(dim,dim).cuda()
        covariance_inv = cov_inv_scaling * torch.eye(dim).cuda()
        samps = torch.tensor(samps)
        samps = samps.cuda()

        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if rank > 1: u = torch.Tensor(u).t().cuda()
            else: u = u.view(-1, 1)
            
            # get determinantal contribution (matrix determinant lemma)
            if rank > 1:
                norm = torch.abs(torch.det(u.t() @ covariance_inv @ u))
            else:
                norm = torch.abs(u.t() @ covariance_inv @ u)

            ideal_rate = (k - len(inds))/(len(samps) - (i))
            # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
            covariance = (i/(i+1))*covariance + (1/(i+1))*(u @ u.t())

            self.zeta = (ideal_rate/(torch.trace(covariance @ covariance_inv))).item()

            pu = np.abs(self.zeta) * norm

            if np.random.rand() < pu.item():
                inds.append(i)
                if early_stop and len(inds) >= k:
                    break
                
                # woodbury update to covariance_inv
                inner_inv = torch.inverse(torch.eye(rank).cuda() + u.t() @ covariance_inv @ u)
                inner_inv = self.inf_replace(inner_inv)
                covariance_inv = covariance_inv - covariance_inv @ u @ inner_inv @ u.t() @ covariance_inv
            else:
                skipped_inds.append(i)

        return inds, skipped_inds


    def get_valid_candidates(self):
        skipped = np.zeros(self.n_pool, dtype=bool)
        skipped[self.skipped] = True
        if self.args["single_pass"]:
            valid = ~self.idxs_lb & ~skipped & self.allowed 
        else:
            valid = ~self.idxs_lb 
        return valid 


    def query(self, n):#, num_round=0):

        valid = self.get_valid_candidates()
        idxs_unlabeled = np.arange(self.n_pool)[valid]

        rank = self.args["rank"]
        if self.args["embs"] == "penultimate":
            gradEmbedding = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
            # print('pen embedding shape: {}'.format(gradEmbedding.shape))
        else:
            gradEmbedding = self.get_exp_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], rank=rank).numpy()
            # print('gradient embedding shape: {}'.format(gradEmbedding.shape))

        early_stop = self.args["early_stop"] 
        cov_inv_scaling = self.args["cov_inv_scaling"]
       
        start_time = time.time()
        chosen, skipped = self.streaming_sampler(gradEmbedding, n, early_stop=early_stop, \
            cov_inv_scaling=cov_inv_scaling, embs = self.args["embs"])
        print(len(idxs_unlabeled), len(chosen), flush=True)
        print('compute time (sec):', time.time() - start_time, flush=True)
        print('chosen: {}, skipped: {}, n:{}'.format(len(chosen),len(skipped),n), flush=True)

        # If more than n samples were selected, take the first n.
        if len(chosen) > n:
            chosen = chosen[:n]

        self.skipped.extend(idxs_unlabeled[skipped])

        result = idxs_unlabeled[chosen]
        if self.args["fill_random"]:
            # If less than n samples where selected, fill is with random samples.
            if len(chosen) < n:
                labelled = np.copy(self.idxs_lb)
                labelled[idxs_unlabeled[chosen]] = True
                remaining_unlabelled = np.arange(self.n_pool)[~labelled]
                n_random = n - len(chosen)
                fillers = remaining_unlabelled[np.random.permutation(len(remaining_unlabelled))][:n_random]
                result = np.concatenate([idxs_unlabeled[chosen], fillers], axis=0)

        return result
