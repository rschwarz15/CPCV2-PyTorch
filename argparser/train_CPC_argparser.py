import argparse
import torch

def argparser():
    parser = argparse.ArgumentParser(description="Training CPC")

    # optional
    parser.add_argument('--dataset',          type=str,   metavar='', default="stl10",    help="Dataset to Use (stl10, cifar10, cifar100)")
    parser.add_argument('--epochs',           type=int,   metavar='', default=10,         help="Number of Epochs for Training")
    parser.add_argument('--trained_epochs',   type=int,   metavar='', default=0,          help="Number of epochs already trained, will load from TrainedModels")
    parser.add_argument('--batch_size',       type=int,   metavar='', default=16,         help="Batch Size")
    parser.add_argument('--lr',               type=float, metavar='', default=2e-4,       help="Learning Rate")
    parser.add_argument('--pred_steps',       type=int,   metavar='', default=5,          help="Number of Predictions Steps")
    parser.add_argument('--neg_samples',      type=int,   metavar='', default=16,         help="Number of Negative Samples for InfoNCE Loss")
    parser.add_argument('--patch_size',       type=int,   metavar='', default=16,         help="Dimension of Patch")
    parser.add_argument('--encoder',          type=str,   metavar='', default="resnet18", help="Which encoder to use (resnet18/34/50/101/152, wideresnet-depth-width, mobilenetV2)")
    parser.add_argument('--norm',             type=str,   metavar='', default="none",     help="What normalisation layer to use (none, batch, layer)")
    parser.add_argument('--print_option',     type=int,   metavar='', default=0,          help="How results are displayed whilst training (0=tqdm, 1=interval statistics, other=End of Epoch only)")
    parser.add_argument('--print_interval',   type=int,   metavar='', default=500,        help="When print_option = 1, this determines how often to print")
    
    parser.add_argument('--download_dataset', action='store_true',                        help="Download the chosen dataset")

    args = parser.parse_args()

    # Add to args given the input choices
    if args.dataset == "stl10":
        args.num_classes = 10
    elif args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    else:
        raise Exception("Invalid Dataset Input")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check encoder choice
    if args.encoder not in ("resnet18", "resnet34", "resnet50", "resent101", "resnet152", "mobilenetV2") and args.encoder[:10] != "wideresnet":
        raise Exception("Invalid Encoder Input")

    return args
    