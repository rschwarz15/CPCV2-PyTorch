import argparse
import torch

def argparser():
    parser = argparse.ArgumentParser(description="Training Classifier")

    # optional
    parser.add_argument('--dataset',             type=str,   metavar='', default="stl10",    help="Dataset to Use (stl10, cifar10, cifar100")
    parser.add_argument('--epochs',              type=int,   metavar='', default=60,         help="Number of Epochs for Training")
    parser.add_argument('--batch_size',          type=int,   metavar='', default=100,        help="Batch Size")
    parser.add_argument('--lr',                  type=float, metavar='', default=0.1,        help="Learning Rate")
    parser.add_argument('--scheduler_step_size', type=int,   metavar='', default=30,         help="Schedular Step Size")
    parser.add_argument('--train_selection',     type=int,   metavar='', default=0,          help="0 = CPC, 1 = Fully Supervised")
    parser.add_argument('--encoder',             type=str,   metavar='', default="resnet34", help="Which encoder to use (resnet34, resnet50, mobilenetV2)")
    parser.add_argument('--model_num',           type=int,   metavar='', default=-1,         help="Number of Epochs that CPC Encoder was trained for (required for CPC classification training)")
    parser.add_argument('--download_dataset',    action='store_true',    default=0,          help="Download the chosen dataset")
    
    args = parser.parse_args()

    # Add to args given the input choices
    if args.dataset == "stl10":
        args.num_classes, args.patch_size = 10, 16
    elif args.dataset == "cifar10":
        args.num_classes, args.patch_size = 10, 8
        raise NotImplementedError
    elif args.dataset == "cifar100":
        args.num_classes, args.patch_size = 100, 8
        raise NotImplementedError
    else:
        raise Exception("Invalid Argument")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
