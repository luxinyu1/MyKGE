import numpy as np
import msgpack
from utils.meta import kb_metadata
from utils.paths import get_dataset_plain_dir, get_dataset_multi_dir
import os

def load_kb_file(kb_file_path):
    with open(kb_file_path, "rb") as f:  
        kb_file = msgpack.unpack(f, encoding="utf-8")
    np_kb_file = np.array(kb_file, dtype=np.int32)
    return np_kb_file

def load_kb_metadata_multi(kb_name):
    try:
        return kb_metadata[kb_name]
    except KeyError:
        raise KeyError("No KB named "+str(kb_name))
        return

def compute_statistics(kb_name, file_to_read="train.kb"):
    nb_rel = load_kb_metadata_multi(kb_name)[1]
    kb_directory = get_dataset_multi_dir(kb_name)
    path_to_file = os.path.join(kb_directory, file_to_read)
    with open(path_to_file, "rb") as f:
        train_triples = msgpack.unpack(f, encoding="utf-8")
    stats = np.array([0] * nb_rel, dtype=np.float32)
    for fact in train_triples:
        stats[fact[0]] += 1
    normalised_stats = np.expand_dims(stats / sum(stats) * nb_rel, axis=-1)
    # 返回norm过后的relation频率分布
    return normalised_stats

def get_dicts(kb_name, entity_dict_name='entities.dict', relation_dict_name='relations.dict'):

    with open(get_dataset_plain_dir(kb_name) / entity_dict_name) as fin:
        entity2id = {}
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(get_dataset_plain_dir(kb_name) / relation_dict_name) as fin:
        relation2id = {}
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    return entity2id, relation2id

def read_triples(kb_name, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    train_triples = []
    test_triples = []
    valid_triples = []
    kb_dir = get_dataset_plain_dir(kb_name)
    for split in ['train', 'test', 'valid']:
        with open(kb_dir / (split+".txt")) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                eval(split+"_triples").append((entity2id[h], relation2id[r], entity2id[t]))
                
    return train_triples, test_triples, valid_triples