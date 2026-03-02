import os
import torch
import argparse
from runner import run
from utils import data_write_csv, setup_seed
from dataloader import get_data

dataPath = "../DEFT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train')

    # parser.add_argument('--path', default=os.path.join(dataPath, 'C:/Users/35272/Desktop/IMv_Project-camera/dataset/'), type=str)
    parser.add_argument('--path', type=str,
                        default=os.path.join(dataPath, './dataset/'))
    # Training info
    parser.add_argument('--data_name', type=list, default=['Reuter', 'GRZA02', 'LandUse_21'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--miss_rate', default=0., type=float)
    parser.add_argument('--patience', type=int, default=10, metavar='LR',
                        help='parameter of Earlystopping [default: 30]')
    parser.add_argument("--teacher_epochs", default=100)
    parser.add_argument('--kd_lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument("--kd_epochs", default=100, type=int)
    parser.add_argument("--lr_factor", type=float, default=0.9)
    parser.add_argument("--lr_patience", type=int, default=5)

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--label_embedd", type=int, default=512)

    # Algorithm hyperparameters
    parser.add_argument('--eta', default=0.05, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lambda_epochs', default=50, type=int)
    parser.add_argument('--ours_start_step', default=0, type=int)
    parser.add_argument('--lam', default=10, type=float, help="balance factor of kd_loss")
    parser.add_argument('--use_rl_eta', action='store_true', default=False)
    parser.add_argument('--ood', default=False, type=bool)  # If use ood strategy

    # args, unknown_args = parser.parse_known_args()
    args, unknown_args = parser.parse_known_args()

    # NOTE：Make uniform dataset for all compared datasets
    # Make uniform dataset
    setup_seed(42)
    miss_rate = [0.0]
    data_name = ['Scene15']

    for dn in data_name:
        args.data_name = dn
        for mr in miss_rate:
            args.miss_rate = mr

            print(f"Processing dataset: {args.data_name}, miss_rate: {args.miss_rate}")

            try:
                X_train, Y_train, X_test, Y_test, Sn_train, Sn_test, dims, class_num = get_data(
                    args.path,
                    args.data_name,
                    use_mean=True)

                print(f"Data loaded successfully: X_train shapes: {[x.shape for x in X_train]}, "
                      f"Y_train shape: {Y_train.shape}, Class num: {class_num}")
                server_dir = f'./MyDataset/{args.data_name}'
                if not os.path.exists(server_dir):
                    os.makedirs(server_dir)

                torch.save(X_train, f'{server_dir}/X_train_{args.miss_rate}.pt')
                torch.save(X_test, f'{server_dir}/X_test_{args.miss_rate}.pt')
                torch.save(Y_train, f'{server_dir}/Y_train_{args.miss_rate}.pt')
                torch.save(Y_test, f'{server_dir}/Y_test_{args.miss_rate}.pt')
                torch.save(Sn_train, f'{server_dir}/Sn_train_{args.miss_rate}.pt')
                torch.save(Sn_test, f'{server_dir}/Sn_test_{args.miss_rate}.pt')
                print(f"Data saved to server path: {server_dir}")

                # NOTE: for other methods
                local_dir = f'./MyDataset/{args.data_name}'
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                torch.save(X_train, f'{local_dir}/X_train_{args.miss_rate}.pt')
                torch.save(X_test, f'{local_dir}/X_test_{args.miss_rate}.pt')
                torch.save(Y_train, f'{local_dir}/Y_train_{args.miss_rate}.pt')
                torch.save(Y_test, f'{local_dir}/Y_test_{args.miss_rate}.pt')
                torch.save(Sn_train, f'{local_dir}/Sn_train_{args.miss_rate}.pt')
                torch.save(Sn_test, f'{local_dir}/Sn_test_{args.miss_rate}.pt')
                print(f"Data saved to local path: {local_dir}")

            except Exception as e:
                print(f"Error processing {args.data_name}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue