import argparse
import json
import os
import socket
import subprocess as sb


if __name__ == "__main__":
    
    # Adapt config if running on SageMaker
    # self-discovered by the presence of the resource config file
    sm_config_path = '/opt/ml/input/config/resourceconfig.json'
    if os.path.exists(sm_config_path):
        with open(sm_config_path) as file:
            cluster_config = json.load(file)

        hosts = cluster_config['hosts']
        default_nodes = len(hosts)
        default_node_rank = hosts.index(os.environ.get("SM_CURRENT_HOST"))
        
        # elect a leader for PyTorch DDP
        leader = socket.gethostbyname(hosts[0])
        for host in cluster_config['hosts']:
            print(f'host: {host}, IP: {socket.gethostbyname(host)}')
        #leader = cluster_config['hosts'][0]  # take first machine in the host list
        
        # Set the network interface for inter node communication
        os.environ['NCCL_SOCKET_IFNAME'] = cluster_config['network_interface_name']
        
        
    else:
        # if not on SageMaker, default to single-machine (eg test on Notebook/IDE)
        default_nodes = 1
        default_node_rank = 0
        leader = '127.0.0.1'

  
    """Set DDP & NCCL environment variables
    https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs
    https://github.com/aws/sagemaker-pytorch-training-toolkit/blob/88ca48a831bf4f099d4c57f3c18e0ff92fa2b48c/src/sagemaker_pytorch_container/training.py#L103
    """
    # Disable IB transport and force to use IP sockets by default
    #os.environ['NCCL_IB_DISABLE'] = '1'
    # Set to INFO for more NCCL debugging information
    os.environ['NCCL_DEBUG'] = 'INFO'        
        

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=2, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS")))
    parser.add_argument("--nodes", type=int, default=default_nodes)
    parser.add_argument("--node_rank", type=int, default=default_node_rank)

    
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    
    
    args, _ = parser.parse_known_args()
    
    print("Number of GPU's in the cluster {}".format(args.gpus))
    print("Number of Nodes in the cluster {}".format(args.nodes))
    print("Local node rank is {}".format(args.node_rank))
    print("Master address is {}".format(leader))
    sb.call(['python', '-m', 
             
             # torch cluster config
             'torch.distributed.launch', 
             '--nproc_per_node', str(args.gpus),
             '--nnodes', str(args.nodes),
             '--node_rank', str(args.node_rank),
             '--master_addr', leader,
             '--master_port', '7777',
             
             # training config
             'train_bert.py',
             '--epochs', str(int(args.epochs)),
             '--num_labels', str(int(args.num_labels)),
             '--batch-size', str(int(args.batch_size)),
             '--test-batch-size', str(args.test_batch_size),
             '--lr', str(args.lr),
             '--momentum', str(int(args.momentum)),
             '--seed', str(args.seed),
             '--momentum', str(args.momentum),
             '--log-interval', str(int(args.log_interval)),
             '--backend', str(args.backend),
             '--workers', str(int(args.workers)),
             '--prefetch', str(int(args.prefetch)),
             '--amp', str(args.amp), 
             '--gpus', str(int(args.gpus)),
             '--nodes', str(args.nodes),
             '--node_rank', str(args.node_rank),
             '--model-dir', str(args.model_dir),
             '--data-dir', str(args.data_dir),
             '--test', str(args.test)
       
            ])