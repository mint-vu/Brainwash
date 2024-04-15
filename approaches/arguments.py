import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='pmnist', type=str, required=False,
                        choices=[ 'split_cifar10_100',
                                  'split_cifar100',
                                  'split_cifar100_SC',
                                  'split_mini_imagenet', 
                                  'split_tiny_imagenet'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--approach', default='lrp', type=str, required=False,
                        choices=['afec_ewc', 'ewc', 'rwalk', 'mas'], help='(default=%(default)s)')
    
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=20, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=16, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--rho', default=0.3, type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--eta', default=0.8, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb_emp', default='0', type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=0, type=float, help='groupsparse parameter')

    parser.add_argument('--img', default=0, type=float, help='image id to visualize')

    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--lasttask', type=int, help='(default=%(default)s)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')

    parser.add_argument('--scenario_name', type = str)
    parser.add_argument('--checkpoint', default=None , type = str)
    parser.add_argument('--addnoise', action='store_true')
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--l2normal', action='store_true')
    parser.add_argument('--blend', action='store_true')
    parser.add_argument('--rndnewds', action='store_true')
    parser.add_argument('--newds', action='store_true')
    parser.add_argument('--rndtopknoise', action='store_true')
    parser.add_argument('--init_acc', action='store_true')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--pattern_add', type = str)

    parser.add_argument('--clip', default=100., type=float)
    parser.add_argument('--optim', type=str, default='sgd')
    
    args=parser.parse_args()
    return args

