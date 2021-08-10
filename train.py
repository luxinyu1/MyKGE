import argparse
import logging
import os
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import MyBoxE, MyRotatE
from utils.kb import get_dicts, read_triples
from utils.paths import CHECKPOINTS_DIR
from dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    filename='./logs/training.log',
    filemode='w',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

logger = logging.getLogger("train")

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
    parser.add_argument("--negative-example-num",
                        type=int,
                        default=100,
                        help="Number of Negative Examples per positive example")
    parser.add_argument("--batch-size", 
                        type=int, 
                        default=1024,
                        help="Batch Size to use for Training")
    parser.add_argument("--loss-margin",
                        type=float,
                        required=False,
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
                        help="Log interval in training.")
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
    parser.add_argument('--adversarial-temperature', 
                        default=1.0, 
                        type=float)

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

    parser.add_argument("--gamma",
                        default=12.0, 
                        type=float)
    parser.add_argument('--negative-sample-size', 
                        default=128, 
                        type=int)
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

def test_or_valid(model, triples, all_true_triples, nentity, nrelation, use_cuda, args):

    model.eval()
    
    #Prepare dataloader for evaluation
    test_dataloader_head = DataLoader(
        TestDataset(
            triples, 
            all_true_triples, 
            nentity,
            nrelation, 
            'head-batch'
        ), 
        batch_size=args.test_valid_batch_size,
        num_workers=0, 
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset(
            triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'tail-batch'
        ), 
        batch_size=args.test_valid_batch_size,
        num_workers=0, 
        collate_fn=TestDataset.collate_fn
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

    if args.model_name == 'BoxE':

        model = MyBoxE(args)
        model.train()

    elif args.model_name == 'RotatE':

        use_cuda = False
        
        entity2id, relation2id = get_dicts(args.target_KB)

        nentity = len(entity2id)
        nrelation = len(relation2id)

        model = MyRotatE(args)

        if not args.no_cuda and torch.cuda.is_available():

            use_cuda = True

            model = model.cuda()

        train_triples, test_triples, valid_triples = read_triples(args.target_KB, entity2id, relation2id)

        all_true_triples = train_triples + test_triples + valid_triples

        logger.info("# Train {} | # Test {} | # Valid {}".format(len(train_triples), len(test_triples), len(valid_triples)))

        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=0, # change here to cpu_num//2 if possible
            collate_fn=TrainDataset.collate_fn
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


        model.train()

        all_positive_loss = 0
        all_negative_loss = 0
        all_loss = 0
        
        for epoch in trange(int(args.epochs), desc="Epoch"):

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

                logger.info("Valid @ Epoch {}".format(epoch+1, step+1))

                res = test_or_valid(model, valid_triples, all_true_triples, nentity, nrelation, use_cuda, args)

                logger.info("[valid]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                    res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
                ))

            if (epoch+1) % args.save_interval == 0:

                if not os.path.exists(CHECKPOINTS_DIR):
                    os.makedirs(CHECKPOINTS_DIR)

                torch.save(model, CHECKPOINTS_DIR / 'epoch_{}.pt'.format(epoch+1))

            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail) # 重载train_iterator

        if args.do_test:

            res = test_or_valid(model, test_triples, all_true_triples, nentity, nrelation, use_cuda, args)

            logger.info("[test]: MR:{}, MRR:{}, Hits@1: {}, Hits@3: {}, Hits@10:{}".format(
                res['MR'], res['MRR'], res['HITS@1'], res['HITS@3'], res['HITS@10']
            ))
                
if __name__ == "__main__":
    main()