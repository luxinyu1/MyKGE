from torch.utils.data import Dataset
import numpy as np
import torch

from utils.calc import convert_to_one_hot
from utils.meta import kb_metadata

class TrainDataset(Dataset):
    
    def __init__(self, triples, nentity, nrelation, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        relation = positive_sample[1]

        positive_sample = [positive_sample[0]] + list(positive_sample[2:]) + [relation]
        positive_tile = np.repeat(np.expand_dims(positive_sample, axis=0), self.negative_sample_size, axis=0)

        replacement_choice = np.random.randint(2, size=self.negative_sample_size)
        replacement_choice = convert_to_one_hot(replacement_choice, depth=3)

        replacement_entities = np.random.randint(self.nentity, size=self.negative_sample_size)
        replacement_tile = np.repeat(np.expand_dims(replacement_entities, axis=1), 3, axis=1)

        negative_sample = np.where(np.equal(replacement_choice, 0), positive_tile, replacement_tile)

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, negative_sample

class TestDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, args):
        self.len = len
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args

        try:
            self.max_arity = kb_metadata[self.args.target_KB][2] # 最大元数
        except KeyError:
            raise KeyError("No KB named "+self.args.target_KB)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        samples = self.triples[idx]
        
        relation = samples[1]

        samples = [samples[0]] + list(samples[2:]) + [relation]
        sample_tile = np.repeat(np.expand_dims(samples, axis=0), self.nentity, axis=0)

        corrupt_samples = []

        for i in range(self.max_arity):

            corrupt_sample = sample_tile.copy()
            entity_range = np.arange(self.nentity)
            corrupt_sample[:,i] = entity_range

            corrupt_sample = torch.LongTensor(corrupt_sample)

            corrupt_samples.append(corrupt_sample)

        samples = torch.LongTensor(samples)
        corrupt_samples = torch.stack(corrupt_samples, dim=2)
        
        return samples, corrupt_samples

    @staticmethod
    def collate_fn(data):

        samples = torch.stack([_[0] for _ in data], dim=0)
        corrupt_samples = torch.stack([_[1] for _ in data], dim=0)

        # print(_corrupt_samples.shape)

        # max_arity = len(_corrupt_samples[0])

        # corrupt_samples = []

        # for j in range(max_arity):
        #     one_corrupt = []
        #     for s in _corrupt_samples:
        #         one_corrupt.append(s[j])
        #     corrupt_samples.append(one_corrupt)

        return samples, corrupt_samples

class CrossSampling_TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2) # 随机抽negative_sample_size对头尾实体
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class CrossSampling_TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float() # 取全部triple的第0个数据
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator:

    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __iter__(self):
        return self
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        for data in dataloader:
            yield data
