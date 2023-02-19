import numpy as np
from numpy.random import default_rng
from .vessal import StreamingSampling
import pdb
import os


class StreamingRand(StreamingSampling):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(StreamingRand, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.skipped = []
        self.rng = default_rng()

    def query(self, n):
        valid = self.get_valid_candidates()
        idxs_unlabeled = np.arange(self.n_pool)[valid]
        chosen, skipped = self.streaming_sampler(self.X[idxs_unlabeled], n, self.args["early_stop"], self.args["deterministic"])
        print(len(idxs_unlabeled), len(chosen))
        print('chosen: {}, skipped: {}, n:{}'.format(len(chosen),len(skipped),n))

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

    def streaming_sampler(self, samps, k, early_stop=True, deterministic=False):
        inds = []
        skipped_inds = []
        ideal_rate = k/len(samps)
        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if not deterministic:
                if np.random.rand() < ideal_rate:
                    inds.append(i)
                else:
                    skipped_inds.append(i)
            else:
                if i % int(1/ideal_rate) == 0:
                    inds.append(i)
                else:
                    skipped_inds.append(i)
            if early_stop and len(inds) >= k:
                break

        return inds, skipped_inds
