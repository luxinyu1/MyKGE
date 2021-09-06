import argparse
import logging
import os
import torch

from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import MyBoxE, MyRotatE, MyTransE
from utils.kb import get_dicts, read_triples
from utils.paths import CHECKPOINTS_DIR, LOG_DIR, TENSORBOARD_DIR
from utils.meta import kb_metadata
from dataloader import TrainDataset, TestDataset, CrossSampling_TrainDataset, CrossSampling_TestDataset, BidirectionalOneShotIterator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    filename=LOG_DIR / 'training.log',
    filemode='w',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("train")

SANITY_EPS = 1e-8

def parse_args(parser):

    parser.add_argument("--model-name",
                        type=str,
                        required=True,
                        help="Model name (RoateE/BoxE)")
    parser.add_argument("--target-KB", 
                        type=str,
                        required=True,
                        help="The Knowledge Base on which to train")
    parser.add_argument("--epochs",
                        type=int, 
                        default=100000, 
                        metavar='', 
                        help="Maximum Number of Epochs to run")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-4,
                        help="Learning Rate to use for training")
    parser.add_argument('--negative-sample-size', 
                        default=128, 
                        type=int,
                        help="Number of Negative Examples per positive example")
    parser.add_argument("--batch-size", 
                        type=int, 
                        default=1024,
                        help="Batch Size to use for Training")
    parser.add_argument("--loss-margin",
                        type=float,
                        required=False,
                        default=3.0,
                        help="The maximum negative distance to consider")
    parser.add_argument("--embedding-dim",
                        type=int,
                        required=True,
                        help="Embedding Dimensionality for points and boxes")
    parser.add_argument("--use-tensorboard",
                        action='store_true',
                        help="Enable Use of TensorBoard during training")
    parser.add_argument("--no-cuda",
                        default=False,
                        action='store_true')
    parser.add_argument("--log-interval",
                        type=int,
                        default=100,
                        help="Log interval(Step) in training.")
    parser.add_argument("--valid-interval",
                        type=int,
                        default=10,
                        help="Validation interval(epoch) in training.")
    parser.add_argument("--save-interval",
                        type=int,
                        default=1,
                        help="Save interval(epoch) in training.")
    parser.add_argument("--test-valid-batch-size",
                        type=int,
                        default=10, # Add this if possible
                        help="Batch size for test and valid.")
    parser.add_argument("--do-test",
                        default=False,
                        action='store_true')

    parser.add_argument("--negative-adversarial-sampling",
                        default=False,
                        action='store_true')
    parser.add_argument("--neg-sampling-opt",
                        type=str,
                        default='uniform') # TODO: merge this two options
    parser.add_argument('--adversarial-temperature', 
                        default=1.0, 
                        type=float)
    # args for BoxE
    parser.add_argument("--use-bumps",
                        type=bool,
                        default=True,
                        help="Allow pairwise bumping to occur, to prevent all-pair correctness.")
    parser.add_argument("--normed-bumps", 
                        default=False,
                        action='store_true',
                        help="Restrict all bumps to be of L2 norm 1.")
    parser.add_argument("--shared-shape",
                        default=False,
                        action='store_true',
                        help="Specifies whether shape is shared by all boxes during training.")
    parser.add_argument("--learnable-shape",
                        type=bool,
                        default=True,
                        help="Specifies whether shape is learned during training.")
    parser.add_argument("--bounded-norm",
                        default=False,
                        action='store_true',
                        help="Limit boxes (following bumps and all processing in the unbounded space) to a minimum "
                            "and maximum size per dimension")
    parser.add_argument("--bound-scale",
                        type=float,
                        default=1.0,
                        help="The finite bounds of the space (if bounded)")
    parser.add_argument("--bounded-pt-space",
                        type=bool,
                        default=True,
                        help="Limit points (following bumps and all processing in the unbounded space) to be mapped to "
                            "the bounded tanh ]-1,1[ space")
    parser.add_argument("--bounded-box",
                        type=bool,
                        default=True,
                        help="")
    parser.add_argument("--fixed-width", 
                        default=False,
                        action='store_true', 
                        help="Specifies whether box width (size) is learned during training")
    parser.add_argument("--total-log-box-size",
                        type=float, 
                        default=-5,
                        help="The total log box size to target during training.")
    parser.add_argument("--hard-total-size",
                        default=False,
                        action='store_true',
                        help="Use Softmax to enforce that all boxes share a hard total size.")
    parser.add_argument("--hard-code-size",
                        default=False,
                        action='store_true',
                        help="Hard Code Size based on statistical appearances of relations in set (works only "
                            "with shared shape)")
    parser.add_argument("--rule-dir", 
                        type=str, 
                        default=False,
                        help="Specify the txt file to read rules from (default no)")
    parser.add_argument("--reg-lambda",
                        type=float,
                        default=0,
                        help="The weight of L2 regularization over bound width (BOX model) to apply")
    parser.add_argument("--reg-points",
                        type=float,
                        default=0,
                        help="Regularisation factor to apply to batch to prevent excessive divergence from center")
    parser.add_argument("--obj-fct",
                        type=str,
                        default='neg_samp')
    parser.add_argument("--gamma",
                        default=12.0, 
                        type=float)
    parser.add_argument("--loss-fct",
                        default='poly',
                        type=str)
    parser.add_argument("--loss-norm-order",
                        type=int,
                        default=2)
    parser.add_argument("--dropout",
                        type=float,
                        default=0.0)

    parser.add_argument('--warm-up-steps', 
                        default=None, 
                        type=int)
    parser.add_argument('--max-steps', 
                        default=100000, 
                        type=int)
    parser.add_argument('--uni-weight', 
                        action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    args = parser.parse_args()

    return args

def test_or_valid(model, triples, nentity, nrelation, use_cuda, args):

    model.eval()
    
    test_dataloader = DataLoader(
        TestDataset(
            triples,
            nentity,
            nrelation,
            args
        ),
        batch_size=args.test_valid_batch_size,
        num_workers=0,
        collate_fn=TestDataset.collate_fn
    )

    logs = []

    with torch.no_grad():
        for samples, corrupt_samples in test_dataloader:

            # Notes: torch.unbind remove the extra dim after spliting the origin tensor

            _corrupt_samples = torch.unbind(corrupt_samples, dim=3)

            __samples = torch.chunk(samples, samples.shape[0], dim=0)

            for pos, corrupt_batch in enumerate(_corrupt_samples):

                # TODO: fake batch here, remove the for loop in the future
                
                __corrupt_samples = torch.unbind(corrupt_batch, dim=0)

                for sample, corrupt_sample in zip(__samples, __corrupt_samples):

                    if use_cuda:
                        corrupt_sample = corrupt_sample.cuda()
                        sample = sample.cuda()

                    score = model(corrupt_sample)

                    indices = torch.argsort(score, dim=0, descending=False)
                    rank = (indices == sample[:,pos].repeat(indices.shape[0])).nonzero()
                    rank = rank.item() + 1

                    logs.append({
                        'MRR': 1.0/rank,
                        'MR': float(rank),
                        'HITS@1': 1.0 if rank <= 1 else 0.0,
                        'HITS@3': 1.0 if rank <= 3 else 0.0,
                        'HITS@10': 1.0 if rank <= 10 else 0.0,
                    })

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    
    return metrics


def crosssampling_test_or_valid(model, triples, all_true_triples, nentity, nrelation, use_cuda, args):

    model.eval()
    
    #Prepare dataloader for evaluation
    test_dataloader_head = DataLoader(
        CrossSampling_TestDataset(
            triples, 
            all_true_triples, 
            nentity,
            nrelation, 
            'head-batch'
        ), 
        batch_size=args.test_valid_batch_size,
        num_workers=0, 
        collate_fn=CrossSampling_TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        CrossSampling_TestDataset(
            triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'tail-batch'
        ), 
        batch_size=args.test_valid_batch_size,
        num_workers=0, 
        collate_fn=CrossSampling_TestDataset.collate_fn
    )
    
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]
    
    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                if use_cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), mode)
                score += filter_bias

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

    return metrics

def main():

    # cpu_num = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    logger.info(args)

    if args.use_tensorboard:

        writer = SummaryWriter(TENSORBOARD_DIR)

    if args.model_name == 'TransE':

        try:
            nentity = kb_metadata[args.target_KB][0]
            nrelation = kb_metadata[args.target_KB][1]
            max_arity = kb_metadata[args.target_KB][2]
        except KeyError:
            raise KeyError("No KB named "+args.target_KB)

        if max_arity > 2:
            raise KeyError("TransE doesn't support kb contains arity > 2.")

        entity2id, relation2id = get_dicts(args.target_KB)

        train_triples, test_triples, valid_triples = read_triples(args.target_KB, entity2id, relation2id)

        train_dataloader = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=TrainDataset.collate_fn
        )

        model = MyTransE(args)

        if args.use_tensorboard:
            graph_input = torch.randint(0, model.relations_num, (1, model.max_arity))
            writer.add_graph(model, graph_input)
            writer.close()

        if not args.no_cuda and torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
        else:
            use_cuda = False
    
        current_learning_rate = args.lr

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )

        loss_margin = torch.tensor(args.loss_margin)

        all_loss = 0.0

        for epoch in trange(int(args.epochs), desc="Epoch"):

            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                positive_samples, negative_samples = batch
                
                optimizer.zero_grad()

                negative_samples = negative_samples.flatten(start_dim=0, end_dim=1)

                if use_cuda:

                    positive_samples = positive_samples.cuda()
                    negative_samples = negative_samples.cuda()

                positive_score = model(positive_samples)
                negative_score = model(negative_samples)

                negative_score = negative_score.reshape(positive_samples.shape[0], args.negative_sample_size)

                negative_score = torch.mean(negative_score, dim=1)

                loss = torch.sum(torch.max(positive_score - negative_score, -loss_margin) + loss_margin)

                all_loss += loss.item()

                loss.backward()
                optimizer.step()

                if (step+1) % args.log_interval == 0:
                    avg_loss = all_loss / args.log_interval
                    logger.info("[Epoch {}] @ Step {} Avg batch loss: {}".format(epoch+1, step+1, avg_loss))
                    all_loss = 0
                
            if (epoch+1) % args.valid_interval == 0:

                logger.info("Valid @ Epoch {}".format(epoch+1))

                res = test_or_valid(model, valid_triples, nentity, nrelation, use_cuda, args)

                logger.info("[valid]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                    res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
                ))

            if (epoch+1) % args.save_interval == 0 and epoch:

                if not os.path.exists(CHECKPOINTS_DIR):
                    os.makedirs(CHECKPOINTS_DIR)

                torch.save(model, CHECKPOINTS_DIR / 'epoch_{}.pt'.format(epoch+1))
        
        if args.do_test:

            res = test_or_valid(model, valid_triples, nentity, nrelation, use_cuda, args)

            logger.info("[test]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
            ))
        

    elif args.model_name == 'BoxE':

        try:
            nentity = kb_metadata[args.target_KB][0]
            nrelation = kb_metadata[args.target_KB][1]
            max_arity = kb_metadata[args.target_KB][2]
        except KeyError:
            raise KeyError("No KB named "+args.target_KB)

        entity2id, relation2id = get_dicts(args.target_KB)

        train_triples, test_triples, valid_triples = read_triples(args.target_KB, entity2id, relation2id)

        train_dataloader = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=TrainDataset.collate_fn
        )

        model = MyBoxE(args)

        if args.use_tensorboard:
            graph_input = torch.randint(0, model.relations_num, (1, model.max_arity))
            writer.add_graph(model, graph_input)
            writer.close()

        if not args.no_cuda and torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
        else:
            use_cuda = False
    
        current_learning_rate = args.lr

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )

        all_loss = 0.0

        for epoch in trange(int(args.epochs), desc="Epoch"):

            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                positive_samples, negative_samples = batch
                
                optimizer.zero_grad()

                negative_samples = negative_samples.flatten(start_dim=0, end_dim=1)

                if use_cuda:

                    positive_samples = positive_samples.cuda()
                    negative_samples = negative_samples.cuda()

                positive_loss = model(positive_samples)
                negative_loss = model(negative_samples)

                if args.obj_fct == 'neg_samp':
                    loss_pos = torch.log(torch.sigmoid(args.loss_margin - positive_loss) + SANITY_EPS)
                elif args.obj_fct == 'margin_based':
                    loss_pos = positive_loss

                if args.neg_sampling_opt == 'uniform':
                    if args.obj_fct == 'neg_samp':  # Standard Objective
                        loss_neg = torch.log(torch.sigmoid(negative_loss - args.loss_margin) + SANITY_EPS)
                        loss_n_term = torch.sum(loss_neg) / args.negative_sample_size
                    elif args.obj_fct == 'margin_based':   # Objective used in TransE
                        reshaped_neg_dists = torch.reshape(negative_loss, [args.negative_sample_size,
                                                                                    args.batch_size])
                        reshaped_neg_dists = torch.transpose(reshaped_neg_dists)
                        loss_neg = torch.mean(reshaped_neg_dists, axis=1)
                        loss_n_term = torch.sum(loss_neg)

                elif args.neg_sampling_opt == 'selfadv':

                    reshaped_neg_dists = torch.reshape(negative_loss, [args.negative_sample_size,
                                                                                args.batch_size])
                    reshaped_neg_dists = torch.transpose(reshaped_neg_dists, perm=[1, 0],
                                                            name='transposed_neg', conjugate=False)
                    softmax_pre_scores = torch.negative(reshaped_neg_dists) * args.adversarial_temperature

                    neg_softmax = torch.nn.softmax(softmax_pre_scores, axis=1)
                    if args.obj_fct == 'neg_samp':
                        loss_neg_batch = torch.log(torch.sigmoid(reshaped_neg_dists - args.margin) + SANITY_EPS)
                        loss_neg = torch.multiply(neg_softmax, loss_neg_batch)
                    elif args.obj_fct == 'margin_based':
                        loss_neg = torch.multiply(neg_softmax, reshaped_neg_dists)
                    loss_n_term = torch.sum(loss_neg)

                loss_p_term = torch.sum(loss_pos)

                # if args.reg_lambda > 0 and not args.hard_total_size:
                #     if args.fixed_width:
                #         logger.info("Box size regularization with fixed widths is redundant, so regularization has been disabled")
                #         reg_lambda = -1
                #         reg_loss = 0.0
                #     else:
                #         reg_loss = total_box_size_reg(rel_deltas=model.rel_deltas, reg_lambda=args.reg_lambda,
                #                                         log_box_size=args.total_log_box_size)
                # else:
                reg_loss = 0.0
                if args.obj_fct == 'neg_samp':
                    loss = - loss_n_term - loss_p_term + reg_loss
                elif args.obj_fct == 'margin_based':
                    loss = torch.reduce_sum(torch.max(0.0, args.loss_margin + loss_pos - loss_neg))
                else:
                    raise ValueError("Error obj fct name.")

                # if args.reg_points > 0:
                #     loss += args.reg_points * (torch.nn.MSELoss(batch_point_representations) +
                #                                     torch.nn.MSELoss(batch_rel_bases))

                all_loss += loss.item()

                loss.backward()
                optimizer.step()

                if (step+1) % args.log_interval == 0:
                    avg_loss = all_loss / args.log_interval
                    logger.info("[Epoch {}] @ Step {} Avg batch loss: {}".format(epoch+1, step+1, avg_loss))
                    all_loss = 0
                
            if (epoch+1) % args.valid_interval == 0:

                logger.info("Valid @ Epoch {}".format(epoch+1))

                res = test_or_valid(model, valid_triples, nentity, nrelation, use_cuda, args)

                logger.info("[valid]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                    res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
                ))

            if (epoch+1) % args.save_interval == 0 and epoch:

                if not os.path.exists(CHECKPOINTS_DIR):
                    os.makedirs(CHECKPOINTS_DIR)

                torch.save(model, CHECKPOINTS_DIR / 'epoch_{}.pt'.format(epoch+1))
        
        if args.do_test:

            res = test_or_valid(model, valid_triples, nentity, nrelation, use_cuda, args)

            logger.info("[test]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
            ))

    elif args.model_name == 'RotatE':
        
        entity2id, relation2id = get_dicts(args.target_KB)

        nentity = len(entity2id)
        nrelation = len(relation2id)

        model = MyRotatE(args)

        if not args.no_cuda and torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
        else:
            use_cuda = False

        train_triples, test_triples, valid_triples = read_triples(args.target_KB, entity2id, relation2id)

        all_true_triples = train_triples + test_triples + valid_triples

        logger.info("# Train {} | # Test {} | # Valid {}".format(len(train_triples), len(test_triples), len(valid_triples)))

        train_dataloader_head = DataLoader(
            CrossSampling_TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=CrossSampling_TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            CrossSampling_TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=CrossSampling_TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.lr

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        logging.info('Start Training...')
        logger.info('lr = %f' % current_learning_rate)
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('embedding_dim = %d' % args.embedding_dim)
        logging.info('gamma = %f' % args.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
        if args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

        all_positive_loss = 0
        all_negative_loss = 0
        all_loss = 0
        
        for epoch in trange(int(args.epochs), desc="Epoch"):

            model.train()

            for step, turple in enumerate(train_iterator):

                positive_batch, negative_batch, subsampling_weight, mode = turple

                optimizer.zero_grad()

                if use_cuda:

                    positive_batch = positive_batch.cuda()
                    negative_batch = negative_batch.cuda()
                    subsampling_weight = subsampling_weight.cuda()

                negative_score = model((positive_batch, negative_batch), mode=mode)

                if args.negative_adversarial_sampling:
                    #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                    negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                    * F.logsigmoid(-negative_score)).sum(dim = 1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

                positive_score = model(positive_batch)

                positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

                if args.uni_weight:
                    positive_batch_loss = - positive_score.mean()
                    negative_batch_loss = - negative_score.mean()
                else: # subsampling用于抵消高频relation的影响
                    positive_batch_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                    negative_batch_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

                loss = (positive_batch_loss + negative_batch_loss)/2

                all_positive_loss += positive_batch_loss.item()
                all_negative_loss += negative_batch_loss.item()
                all_loss += loss.item()

                loss.backward()
                
                optimizer.step()

                if (step+1) % args.log_interval == 0:
                    avg_positive_loss = all_positive_loss / args.log_interval
                    avg_negative_loss = all_negative_loss / args.log_interval
                    avg_loss = all_loss / args.log_interval
                    logger.info("[Epoch {}] @ Step {} Avg positive batch loss: {} Avg negative batch loss: {} Avg batch loss: {}".format(epoch+1, step+1, avg_positive_loss, avg_negative_loss, avg_loss))
                    all_positive_loss = 0
                    all_negative_loss = 0
                    all_loss = 0
                
            if (epoch+1) % args.valid_interval == 0:

                logger.info("Valid @ Epoch {}".format(epoch+1))

                res = crosssampling_test_or_valid(model, valid_triples, all_true_triples, nentity, nrelation, use_cuda, args)

                logger.info("[valid]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                    res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
                ))

            if (epoch+1) % args.save_interval == 0:

                if not os.path.exists(CHECKPOINTS_DIR):
                    os.makedirs(CHECKPOINTS_DIR)

                torch.save(model, CHECKPOINTS_DIR / 'epoch_{}.pt'.format(epoch+1))

            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail) # 重载train_iterator

        if args.do_test:

            res = crosssampling_test_or_valid(model, test_triples, all_true_triples, nentity, nrelation, use_cuda, args)

            logger.info("[test]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
            ))
    else:
        raise KeyError("Model has not been implemented. Check model name again.")
                
if __name__ == "__main__":
    main()