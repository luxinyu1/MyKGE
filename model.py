import numpy as np
import logging
import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset

from utils.kb import load_kb_file, compute_statistics
from utils.paths import get_dataset_dir
from utils.meta import kb_metadata

SANITY_EPS = 1e-8
NORM_LOG_BOUND = 1

zero = torch.tensor(0.0)
half = torch.tensor(0.5)
one = torch.tensor(1.0)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    filename='./logs/training.log',
    filemode='w',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("model")

def compute_box(box_base, box_delta):
    box_second = box_base + half * box_delta
    box_first = box_base - half * box_delta
    box_low = torch.min(box_first, box_second)
    box_high = torch.max(box_first, box_second)
    return box_low, box_high

def init_var(shape, min_val, max_val):
    var = nn.Parameter(torch.zeros(shape))
    nn.init.uniform_(
        tensor=var, 
        a=min_val, 
        b=max_val
    )
    return var

# TODO: Why normalise like this?
def product_normalise(input_tensor, bounded_norm=True):
    step1_tensor = torch.abs(input_tensor)
    step2_tensor = step1_tensor + SANITY_EPS
    log_norm_tensor = torch.log(step2_tensor)
    step3_tensor = torch.mean(log_norm_tensor, dim=2, keepdims=True)
    norm_volume = torch.exp(step3_tensor)
    pre_norm_out = input_tensor / norm_volume
    if not bounded_norm:

        return pre_norm_out

    else:

        minsize_tensor = torch.minimum(torch.min(log_norm_tensor, dim=2, keepdim=True), -NORM_LOG_BOUND)
        maxsize_tensor = torch.maximum(torch.max(log_norm_tensor, dim=2, keepdim=True), NORM_LOG_BOUND)
        minsize_ratio = -NORM_LOG_BOUND / minsize_tensor
        maxsize_ratio = NORM_LOG_BOUND / maxsize_tensor
        size_norm_ratio = torch.minimum(minsize_ratio, maxsize_ratio)
        normed_tensor = log_norm_tensor * size_norm_ratio

        return torch.exp(normed_tensor)

def instantiate_box_embeddings(scale_mult_shape, rel_tbl_shape, base_norm_shapes, sqrt_dim,
                               hard_size: bool, total_size: float, relation_stats, fixed_width: bool):
    if relation_stats is not None:
        # [relations_num, max_arity, 1]
        scale_multiples = relation_stats
    else:
        if fixed_width:
            scale_multiples = torch.zeros(scale_mult_shape)
        else:
            scale_multiples = init_var(scale_mult_shape, -1.0, 1.0)
        if hard_size:
            scale_multiples = total_size * F.softmax(scale_multiples, dim=0)
        else:
            scale_multiples = F.elu(scale_multiples) + one

    # box_mbedding 分为 embedding_base_points, embedding_deltas
    # [relations_num, max_arity, embedding_dim]
    embedding_base_points = init_var(rel_tbl_shape, -0.5 / sqrt_dim, 0.5 / sqrt_dim)
    # [relations_num, max_arity, embedding_dim]
    embedding_deltas = torch.mul(scale_multiples, base_norm_shapes)
    return embedding_base_points, embedding_deltas, scale_multiples

def add_padding(input_tensor):
    return torch.cat([input_tensor, torch.zeros([1, input_tensor.shape[1]])], dim=0)

def q2b_loss(points, lower_corner, upper_corner):
    # Query2Box Loss Function: https://arxiv.org/pdf/2002.05969.pdf
    centres = 1 / 2 * (lower_corner + upper_corner)
    dist_outside = torch.max(points - upper_corner, 0.0) + torch.max(lower_corner - points, 0.0)
    dist_inside = centres - torch.min(upper_corner, torch.min(lower_corner, points))
    
    return dist_outside, dist_inside

def loss_function_q2b(batch_points, batch_mask, rel_bx_low, rel_bx_high, batch_rel_mults, dim_dropout_prob=zero, order=2, alpha=0.2):
    batch_box_inside, batch_box_outside = q2b_loss("Q2B_Box_Loss", batch_points, rel_bx_low, rel_bx_high,
                                                batch_rel_mults)
    
    bbi = torch.norm(F.dropout(batch_box_inside, p=dim_dropout_prob), dim=2, p=order)
    bbi_masked = torch.sum(torch.mul(bbi, batch_mask), dim=1)
    bbo = torch.norm(F.dropout(batch_box_outside, p=dim_dropout_prob), dim=2, p=order)
    bbo_masked = torch.sum(torch.mul(bbo, batch_mask), dim=1)
    total_loss = alpha * bbi_masked + bbo_masked

    return total_loss

def polynomial_loss(points, lower_corner, upper_corner):
    widths = upper_corner - lower_corner
    widths_p1 = widths + one # width incremented by 1
    centres = 0.5 * (lower_corner + upper_corner)
    # calculate the distance
    # 在超立方体内, 除以widths_p1, 在超立方体外, 乘widths_p1 - k
    width_cond = torch.where(torch.logical_and(lower_corner <= points, points <= upper_corner),
                            torch.abs(points - centres) / widths_p1,
                            widths_p1 * torch.abs(points - centres) - (widths / 2) * (widths_p1 - 1 / widths_p1))
    return width_cond

def loss_function_poly(batch_points, batch_mask, rel_bx_low, rel_bx_high, batch_rel_mults, dim_dropout_prob=zero, order=1):
    poly_loss = polynomial_loss(batch_points, rel_bx_low, rel_bx_high, batch_rel_mults)
    poly_loss = torch.norm(F.dropout(poly_loss, p=dim_dropout_prob), dim=2, p=order)
    total_loss = torch.sum(torch.mul(poly_loss, batch_mask), dim=1)
    return total_loss

class MyBoxE(nn.Module):

    def __init__(self, args):

        super(MyBoxE, self).__init__() 
        
        self.args = args

        try:
            self.entities_num = kb_metadata[self.args.target_KB][0] # 数据集中不同的实体个数
            self.relations_num = kb_metadata[self.args.target_KB][1] # 数据集中不同的关系个数
            self.max_arity = kb_metadata[self.args.target_KB][2] # 最大元数
        except KeyError:
            raise KeyError("No KB named "+self.args.target_KB)
        
        self.train_dataloader = self.load_training_set()
        self.embed()

    def embed(self):
        # entity embeddings
        entity_table_shape = [self.entities_num, self.args.embedding_dim]
        self.sqrt_dim = torch.sqrt(torch.tensor(self.args.embedding_dim + 0.0))
        # 限定了embeddings的上下界[-1/2*sqrt(embedding_dim), 1/2*sqrt(embedding_dim)]
        self.entity_points = init_var(entity_table_shape, -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
        self.entities_with_pad = add_padding(self.entity_points)
        # translational bumps
        if self.args.use_bumps:
            self.entity_bumps = init_var(entity_table_shape, -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
            if self.args.normed_bumps:  # Normalization of bumps option
                self.entity_bumps = F.normalize(self.entity_bumps, p=2, dim=1)
            self.bumps_with_pad = add_padding(self.entity_bumps)
        # relation embeddings
        relation_table_shape = [self.relations_num, self.max_arity, self.args.embedding_dim]
        scale_multiples_shape = [self.relations_num, self.max_arity, 1]
        tile_shape = [self.relations_num, 1, 1]
        if self.args.hard_code_size:
            relation_stats = compute_statistics(self.args.target_KB)
            relation_stats = torch.tensor(relation_stats ** (1 / self.args.embedding_dim))
        else:
            relation_stats = None
        if self.args.shared_shape: # Shared box shape
            base_shape = [1, self.max_arity, self.args.embedding_dim]
            tile_var = True
        else: # Variable box shape
            base_shape = relation_table_shape
            tile_var = False
        if self.args.learnable_shape: # If shape is learnable, define variables accordingly
            self.rel_shapes = init_var(base_shape, -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
            self.norm_rel_shapes = product_normalise(self.rel_shapes, self.args.bounded_norm) # TODO:why?
        else: # Otherwise set all boxes as one-hypercubes
            self.norm_rel_shapes = torch.ones(base_shape)
        if tile_var:
            # 对[1, max_arity, embedding_dim]平铺复制relations_num次, 最后得到[relations_num, max_arity, embedding_num]
            self.norm_rel_shapes = self.norm_rel_shapes.repeat(tile_shape)
        self.total_size = np.exp(self.args.total_log_box_size) if self.args.hard_total_size else -1

        self.rel_bases, self.rel_deltas, self.rel_multiples = \
            instantiate_box_embeddings(scale_multiples_shape, relation_table_shape,
                                    self.norm_rel_shapes, self.sqrt_dim, self.args.hard_total_size,
                                    self.total_size, relation_stats, self.args.fixed_width)

        if self.args.rule_dir:   # Rule Injection logic
            pass
        else:
            pass


    def load_training_set(self):
        kb_data = load_kb_file(get_dataset_dir(self.args.target_KB) / 'train.kb')
        kb_data = torch.tensor(kb_data)
        # if not options.restricted_training:
        num_training_samples = kb_data.shape[0]
        train_data = TensorDataset(kb_data)
        
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader

    # else:
    #     tr_np_arr = tr_np_arr[:options.restriction, :]
    #     self.nb_training_facts = options.restriction
    # if self.augment_inv:
    #     tr_np_arr_augmentation = np.zeros_like(tr_np_arr)
    #     tr_np_arr_augmentation[:, 0] = tr_np_arr[:, 0] + self.original_nb_rel
    #     tr_np_arr_augmentation[:, 1] = tr_np_arr[:, 2]
    #     tr_np_arr_augmentation[:, 2] = tr_np_arr[:, 1]
    #     tr_np_arr_augmentation[:, 3] = tr_np_arr[:, 3]
    #     tr_np_arr = np.concatenate([tr_np_arr, tr_np_arr_augmentation], axis=0)
    #     self.nb_training_facts = 2 * self.nb_training_facts

    def forward(self):
        # Forward function that calculate the score of a batch of triples.

        if self.reg_lambda > 0 and not self.hard_total_size:
            if self.fixed_width:
                print("Box size regularization with fixed widths is redundant, so regularization has been disabled")
                self.reg_lambda = -1
                self.reg_loss = 0.0
            else:
                self.reg_loss = total_box_size_reg(rel_deltas=self.rel_deltas, reg_lambda=self.reg_lambda,
                                                   log_box_size=self.total_log_box_size)
        else:
            self.reg_loss = 0.0
        if self.obj_fct == Cnst.NEG_SAMP:
            self.loss = - self.loss_n_term - self.loss_p_term + self.reg_loss
        elif self.obj_fct == Cnst.MARGIN_BASED:
            self.loss = tf.reduce_sum(tf.maximum(0.0, self.margin + self.loss_pos - self.loss_neg))

        if self.reg_points > 0:
            self.loss += self.reg_points * (tf.nn.l2_loss(self.batch_point_representations) +
                                            tf.nn.l2_loss(self.batch_rel_bases))

        return score

    def train(self):


        for e in trange(int(self.args.epochs), desc="Epoch"):
            
            average_epoch_loss = 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                pass

class MyRotatE(nn.Module):

    def __init__(self, args):

        super(MyRotatE, self).__init__()

        self.epsilon = 2.0

        self.args = args

        try:
            self.entities_num = kb_metadata[self.args.target_KB][0] # 数据集中不同的实体个数
            self.relations_num = kb_metadata[self.args.target_KB][1] # 数据集中不同的关系个数
        except KeyError:
            raise KeyError("No KB named "+self.args.target_KB)

        self.entity_dim = self.args.embedding_dim * 2 # TODO:意义暂时不明
        # self.relation_dim = self.args.embedding_dim * 2
        self.relation_dim = self.args.embedding_dim

        self.gamma = nn.Parameter(
            torch.Tensor([self.args.gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.args.embedding_dim]), 
            requires_grad=False
        )

        self.entity_embedding = init_var([self.entities_num, self.entity_dim], 
                                        -self.embedding_range.item(), self.embedding_range.item())
        self.relation_embedding = init_var([self.relations_num, self.relation_dim], 
                                        -self.embedding_range.item(), self.embedding_range.item())
        
    def forward(self, sample, mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        
        # Caculate Score

        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)

        return score