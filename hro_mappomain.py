import argparse
from typing import List

import torch

from hro_mappoTrainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser()

    env_group = parser.add_argument_group("Environment and Traffic")
    env_group.add_argument('--cav_num', type=int, default=5, help='Number of CAVs')
    env_group.add_argument(
        '--LANES_LEFT_TO_RIGHT',
        type=List,
        default=['E0_2', 'E0_1', 'E0_0'],
        help='Lane IDs ordered from left to right'
    )

    env_group.add_argument('--graph_node_dim', type=int, default=8, help='Node feature dimension')
    env_group.add_argument('--planning_horizon', type=int, default=20, help='Planning horizon in time steps')
    env_group.add_argument('--T', type=float, default=1.0, help='Simulation base time step in seconds')

    env_group.add_argument('--d_0', type=float, default=4.5, help='Vehicle length in meters')
    env_group.add_argument('--r_0', type=float, default=2.0, help='Minimum safety gap in meters')
    env_group.add_argument('--t_safe', type=float, default=1.0, help='Safety time headway in seconds')
    env_group.add_argument('--v_max', type=float, default=33.0, help='Maximum CAV speed in m/s')
    env_group.add_argument('--v_min', type=float, default=15.0, help='Minimum CAV speed in m/s')
    env_group.add_argument('--a_max', type=float, default=2.0, help='Maximum acceleration')
    env_group.add_argument('--b_max', type=float, default=5.0, help='Maximum deceleration')

    env_group.add_argument('--w_cong', type=float, default=5.0, help='Congestion wave speed in m/s')
    env_group.add_argument('--jam_density', type=float, default=0.09, help='Jam density in veh/m')
    env_group.add_argument('--alpha_right', type=float, default=0.021530, help='HDV right lane-change probability parameter')
    env_group.add_argument('--alpha_left', type=float, default=0.022544, help='HDV left lane-change probability parameter')

    env_group.add_argument('--max_steps', type=int, default=1500, help='Maximum simulation steps per episode')
    env_group.add_argument('--control_interval', type=float, default=1.0, help='Decision interval in seconds')

    env_group.add_argument('--T1', type=float, default=0.56, help='Minimum acceptable HDV headway')
    env_group.add_argument('--T2', type=float, default=2.34, help='Maximum acceptable HDV headway')
    env_group.add_argument('--DT', type=float, default=0.1, help='HDV simulation step length')
    env_group.add_argument('--P', type=float, default=0.015, help='Target headway resampling probability')

    env_group.add_argument('--action_dim_dis', type=int, default=3, help='Discrete action dimension')

    train_group = parser.add_argument_group("Training and Logging")
    train_group.add_argument('--random_seed', type=int, default=1)
    train_group.add_argument(
        '--device',
        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        help='Device'
    )
    train_group.add_argument('--max_episodes', type=int, default=2000)
    train_group.add_argument('--episode_length', type=int, default=150, help='Rollout length in control steps')

    train_group.add_argument('--buffer_size', type=int, default=4096, help='Replay buffer capacity')
    train_group.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')
    train_group.add_argument('--num_mini_batch', type=int, default=4, help='Number of mini-batches per epoch')
    train_group.add_argument('--agent_save_freq', type=int, default=50, help='Model save interval in episodes')
    train_group.add_argument('--agent_update_freq', type=int, default=6, help='Update interval in episodes')
    train_group.add_argument('--agent_update_minsize', type=int, default=1024, help='Minimum buffer size before update')

    train_group.add_argument('--record_mark', type=str, default='renaissance', help='Experiment tag')
    train_group.add_argument('--log_dir', type=str, default='runs/experiment_9_11_1', help='TensorBoard log directory')

    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument('--lr_actor', type=float, default=5e-4)
    optim_group.add_argument('--lr_critic', type=float, default=1e-3)
    optim_group.add_argument('--opti_eps', type=float, default=1e-5, help='Optimizer epsilon')
    optim_group.add_argument('--weight_decay', type=float, default=0.0)

    optim_group.add_argument('--lr_std', type=float, default=5e-4, help=argparse.SUPPRESS)
    optim_group.add_argument('--target_kl_con', type=float, default=0.015, help=argparse.SUPPRESS)

    network_group = parser.add_argument_group("Network")
    network_group.add_argument('--hidden_size', type=int, default=256, help='Hidden size for graph encoder and value heads')
    network_group.add_argument('--layer_N', type=int, default=2, help='Number of graph layers')

    network_group.add_argument('--use_feature_normalization', action='store_true', default=True, help='Use LayerNorm on inputs')
    network_group.add_argument('--use_orthogonal', action='store_true', default=True, help='Use orthogonal initialization')
    network_group.add_argument('--use_ReLU', action='store_true', default=True, help='Use ReLU activation')

    ppo_group = parser.add_argument_group("PPO")
    ppo_group.add_argument('--ppo_epoch', type=int, default=10)
    ppo_group.add_argument('--clip_param', type=float, default=0.25, help='PPO ratio clip')
    ppo_group.add_argument('--entropy_coef', type=float, default=0.02)
    ppo_group.add_argument('--value_loss_coef', type=float, default=0.5)
    ppo_group.add_argument('--max_grad_norm', type=float, default=0.7)
    ppo_group.add_argument('--use_max_grad_norm', action='store_true', default=True)

    ppo_group.add_argument('--use_gae', action='store_true', default=True)
    ppo_group.add_argument('--gamma', type=float, default=0.99)
    ppo_group.add_argument('--gae_lambda', type=float, default=0.95)
    ppo_group.add_argument('--use_clipped_value_loss', action='store_true', default=True)
    ppo_group.add_argument('--use_huber_loss', action='store_true', default=True)
    ppo_group.add_argument('--huber_delta', type=float, default=10.0)
    ppo_group.add_argument('--target_kl_dis', type=float, default=0.015, help='KL early-stop threshold for discrete actions')

    ppo_group.add_argument('--use_policy_active_masks', action='store_true', default=True, help='Mask invalid samples in policy loss')
    ppo_group.add_argument('--use_value_active_masks', action='store_true', default=True, help='Mask invalid samples in value loss')

    ppo_group.add_argument('--use_centralized_V', action='store_true', default=True, help='Use global critic')
    ppo_group.add_argument('--use_proper_time_limits', action='store_true', default=False, help='Handle episode truncation explicitly')
    ppo_group.add_argument('--use_linear_lr_decay', action='store_true', default=True, help='Use linear learning-rate decay')

    ppo_group.add_argument("--gain", type=float, default=0.01, help="Gain of the last action layer")

    graph_group = parser.add_argument_group("Graph Encoder")
    graph_group.add_argument(
        "--conv_type",
        type=str,
        default="gat",
        choices=["gcn", "gat", "gin", "sage"],
        help="GNN convolution type"
    )
    graph_group.add_argument(
        "--use_jk",
        action="store_true",
        help="Enable Jumping Knowledge"
    )
    graph_group.add_argument(
        "--jk_mode",
        type=str,
        default="concat",
        choices=["concat", "max"],
        help="Jumping Knowledge aggregation mode"
    )
    graph_group.add_argument(
        "--gat_heads",
        type=int,
        default=4,
        help="Number of attention heads when conv_type=gat"
    )
    graph_group.add_argument(
        "--gat_concat",
        type=str2bool,
        default=True,
        help="Whether to concatenate GAT heads"
    )

    norm_group = parser.add_argument_group("Value Normalization")
    norm_group.add_argument('--use_popart', action='store_true', default=False, help='Use PopArt for value normalization')
    norm_group.add_argument('--use_valuenorm', action='store_true', default=True, help='Use ValueNorm for value normalization')

    rnn_group = parser.add_argument_group("RNN")
    rnn_group.add_argument('--use_naive_recurrent_policy', action='store_true', default=False)
    rnn_group.add_argument('--use_recurrent_policy', action='store_true', default=False)
    rnn_group.add_argument('--recurrent_N', type=int, default=1)
    rnn_group.add_argument('--data_chunk_length', type=int, default=10)

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--global_coef", type=float, default=0, help="Weight of global critic")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    sumo_binary = "sumo"
    sumo_config = "Env/hplatoon.sumocfg"

    trainer = Trainer(args, sumo_binary, sumo_config)
    trainer.run()


if __name__ == '__main__':
    main()