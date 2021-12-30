import argparse
from optimizer.lr_scheduler import LR_SCHEDULER_REGISTRY


def parse_args_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='')
    parser.add_argument('--g', type=str, default=3)
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model on dev set')
    parser.add_argument('--train-and-eval', action='store_true', help='evaluate the model per epoch during training')
    parser.add_argument('--eval-n-epoch', type=int, default=1, help='')
    parser.add_argument('--load-pretrained-model', action='store_true', help='load pretrained model')
    parser.add_argument('--partial-load', action='store_true', help='load pretrained model for partial modules')

    parser.add_argument('--model-state-dict-path', default='results/model-best', type=str, help='model_state_dict_path')
    parser.add_argument('--model-saved-path', type=str, default='results/', help='')
    parser.add_argument('--mn', type=str, default='', help='model_name')
    parser.add_argument('--dec', type=str, default='rnnt', help='decoder approach')

    parser.add_argument('--dataset', choices=['GRID'], default='GRID', help='')
    parser.add_argument('--model-name', choices=['simul_lr'], default='simul_lr', help='')
    parser.add_argument('--text-level', choices=['word', 'subword', 'char'], default='word', help='')
    parser.add_argument('--pretrain-dim', type=int, default=300, help='')
    parser.add_argument('--chunk-size', type=int, default=99, help='')
    parser.add_argument('--mem-size', type=int, default=0, help='')
    parser.add_argument('--vocab-path', type=str, default='', help='')
    parser.add_argument('--train-data', type=str, default='', help='')
    parser.add_argument('--val-data', type=str, default=None, help='')
    parser.add_argument('--test-data', type=str, default='', help='')
    parser.add_argument('--data-rate', type=float, default=0.3, help='')
    parser.add_argument('--use-word', type=bool, default=True, help='')
    parser.add_argument('--text-max-length', type=int, default=10, help='')
    parser.add_argument('--video-max-length', type=int, default=75, help='')
    parser.add_argument('--d-model', type=int, default=256, help='')
    parser.add_argument('--num-heads', type=int, default=8, help='')
    parser.add_argument('--num-layers', type=int, default=6, help='')
    parser.add_argument('--batch-size', type=int, default=32, help='')
    parser.add_argument('--vocab-size', type=int, default=52 + 3, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--d-ff', type=int, default=512, help='')
    parser.add_argument('--ffn-layer', type=str, default='fc', help='')
    parser.add_argument('--first-kernel-size', type=int, default=1, help='')
    parser.add_argument('--num-gcn-layers', type=int, default=2, help='')
    parser.add_argument('--display-n-batches', type=int, default=200, help='')
    parser.add_argument('--max-num-epochs', type=int, default=150, help='')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD', help='weight decay')
    parser.add_argument('--lr-scheduler', default='cosine', choices=LR_SCHEDULER_REGISTRY.keys(), help='Learning Rate Scheduler')
    parser.add_argument('--lr', default=[1e-4, 2e-4], type=list, help='learning rate')
    parser.add_argument('--max-update', type=int, default=100, help='')
    parser.add_argument('--lr-shrink', type=float, default=0.999, help='')
    # after self-training training main task
    parser.add_argument('--main-task', action='store_true', help='whether to train main task')
    parser.add_argument('--CTC-epoch', default=10, type=int, help='the number of epochs to pretrain by CTC')
    parser.add_argument('--warm-epoch', default=50, type=int, help='the number of epochs to warm-up C3D')

    from optimizer.lr_scheduler.cosine_lr_scheduler import CosineSchedule
    CosineSchedule.add_args(parser)

    from optimizer.adam_optimizer import AdamOptimizer
    AdamOptimizer.add_args(parser)
    return parser.parse_args()