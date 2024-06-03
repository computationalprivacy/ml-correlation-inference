import numpy as np
import pandas as pd


class ArbitraryMarginal(object):
    def __init__(self, bins, cdf, binary):
        self.bins = bins
        self.cdf = cdf
        self.binary = binary
        if binary:
            assert len(self.cdf) == 2
        else:
            assert len(self.cdf) >= 3 and self.cdf[0] == 0 and self.cdf[-1] == 1


    def uniform(self, a, b):
        assert a < b
        u = np.random.random()
        return a + (b-a) * u
    

    def sample(self, nbr_samples):
        samples = []
        for _ in range(nbr_samples):
            random = np.random.random()
            samples.append(self.get_inversion(random))
        return samples

   
    def binary_lookup(self, u):
        if np.abs(u - self.cdf[-1]) < 1e-6:
            return len(self.cdf)-1, self.bins[-1]
        begin, end = 0, len(self.cdf)-1
        #print('Begin', begin, 'End', end)
        while begin < end:
            middle = (begin + end) // 2
            #print('Middle', middle)
            if self.cdf[middle] <= u < self.cdf[middle + 1]:
                #print('I am here 1')
                return middle, np.random.uniform(self.bins[middle], self.bins[middle+1])
            elif begin < end - 1:
                if self.cdf[middle] <= u:
                    #print('I am here 2')
                    begin = middle
                else:
                    end = middle
            #print('Middle', middle, 'Begin', begin, 'End', end)
        return end, self.bins[end]
    

    def get_inversion(self, u):
        assert 0 <= u <= 1
        if self.binary:
            if u < self.cdf[0]:
                return 0
            else:
                return 1
        else:
            return self.binary_lookup(u)[1]
            #for i in range(len(self.cdf)-1):
            #    if self.cdf[i] <= u < self.cdf[i+1]:
            #         return self.uniform(self.bins[i], self.bins[i+1])
            #return self.bins[-1]

    #fonction needed to ensure that copulas are working with our new class
    def percent_point(self, cdf):
        output = []
        for u in cdf:
            output.append(self.get_inversion(u))
        return output


def compute_histogram(column, K, binary=False):
    # The intervals need to be [-inf, min(column)); 
    # [min(column), max(column))/K, [max_column, inf) to match the empirical
    # distribution.
    assert K > 0, f'ERROR: Invalid value for {K}.'
    if isinstance(column, pd.core.series.Series):
        column = column.to_numpy()
    elif isinstance(column, list):
        column = np.array(column)
    else:
        assert isinstance(column, np.ndarray), 'ERROR: Invalid column format.'
    if binary:
        # Check that y is indeed binary.
        c0 = np.sum(column==0)
        c1 = np.sum(column==1)
        assert c0 + c1 == len(column), \
                'ERROR: y is not a binary variable.'
        p0, p1 = c0/len(column), c1/len(column)
        return None, [p0, p0+p1]
    else:
        m, M = np.min(column), np.max(column)
        assert m < M, f'ERROR: This column has only one possible value.'
        bins = np.linspace(m, M, K+1)
        #print(bins)
        hist, _  = np.histogram(column, bins=bins)
        pdf = hist / np.sum(hist)
        #print(bins, pdf)
        #assert np.abs(np.sum(pdf)-1) < 1e-6, pdf
        cdf = [0] + list(np.cumsum(pdf))
        cdf[-1] = 1
        return bins, cdf

