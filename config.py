import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--train_only', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--model', type=str, default='BrainWaveNet')
    parser.add_argument('--main_path', type=str, default='./BrainWaveNet/')

    # data
    parser.add_argument('--data', type=str, default='ABIDE')
    parser.add_argument('--data_path', type=str, default='./ABIDE_I/cc200/cpac_filt_noglobal')
    parser.add_argument('--label_path', type=str, default='./ABIDE_I/Phenotypic_V1_0b_preprocessed1.csv')
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--train_set', type=float, default=0.7)
    parser.add_argument('--valid_set', type=float, default=0.2)
    parser.add_argument('--test_set', type=float, default=0.1)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--mixup_data', type=bool, default=True)

    # model
    parser.add_argument('--front_end', type=bool, default=True)
    parser.add_argument('--in_channels', type=int, default=200)
    parser.add_argument('--out_channels', type=int, default=200)

    parser.add_argument('--n_channels', type=int, default=200)
    parser.add_argument('--n_frequencies', type=int, default=5)
    parser.add_argument('--n_times', type=int, default=78)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--temporal_dmodel', type=int, default=16)
    parser.add_argument('--temporal_nheads', type=int, default=4)
    parser.add_argument('--temporal_dimff', type=int, default=32)

    parser.add_argument('--spatial_dmodel', type=int, default=64)
    parser.add_argument('--spatial_nheads', type=int, default=8)
    parser.add_argument('--spatial_dimff', type=int, default=128)

    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--use_tct', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=2)

    # train
    parser.add_argument('--n_fold', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler_mode', type=str, default='cos')
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--target_lr', type=float, default=1e-6)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--warm_up_from', type=float, default=0.0)
    parser.add_argument('--warm_up_steps', type=int, default=0)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)

    # wadb
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--project', type=str, default='BrainWaveNet')

    args = parser.parse_args()
    return args
