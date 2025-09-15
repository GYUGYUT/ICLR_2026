import torch
import torch.multiprocessing as mp
from train.ddp_train import *
from util.utils import set_seed
import argparse
from util.utils import create_save_directory

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="OS-Prompt Training Script with DDP")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of prompts in the prompt pool")
    parser.add_argument("--prompt_dim", type=int, default=768, help="Dimension of each prompt")
    parser.add_argument("--wandb_project", type=str, default="OS-Prompt", help="WandB project name")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save_criteria", type=str, default="loss", help="Criteria to save the best model")
    parser.add_argument("--scheduler", type=str, choices=["step", "multistep", "cosine", "plateau"], default="step", help="LR Scheduler type")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR and MultiStepLR")
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 60, 90], help="Milestones for MultiStepLR")
    parser.add_argument("--loss_function", type=str, default="cross_entropy", help="Loss function to use")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_mode", type=int, default=0, help="all_train : 0 , step1_train : 1, step2_train : 2")
    parser.add_argument("--step_1_model_pt_path", type=str, default=None, help="step_1_model_pt")
    parser.add_argument("--model_name", type=str, default=None, help="model_name")
    parser.add_argument("--save_path", type=str, default=None, help="save_path")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")

    parser.add_argument("--train_csv_file", type=str, default="None")
    parser.add_argument("--val_csv_file", type=str, default="None")
    parser.add_argument("--test_csv_file", type=str, default="None")
    parser.add_argument("--image_folder", type=str, default="None")
    parser.add_argument("--num_classes", type=int, default=1000)

    parser.add_argument("--diabetic_train_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_val_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_test_csv_file", type=str, default="None")
    parser.add_argument("--diabetic_image_folder", type=str, default="None")
    parser.add_argument("--diabetic_classes", type=int, default=1000)

    

    

    return parser.parse_args()


def main():
    # Argument Parsing
    args = parse_args()

    set_seed(1004)

    gpu_ids = args.gpu_ids.split(",")

    args.save_path = create_save_directory()

    if args.train_mode == 0:

        ddp_train_step1(args)

        ddp_train_step2(args)

        ddp_train_step3(args)




if __name__ == "__main__":
    main()