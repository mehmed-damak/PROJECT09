"""
Script to train H1 robot using LearningHumanoidWalking's full PPO implementation
Run this script to use their advanced PPO with your H1 environment
"""
import sys
import os
from pathlib import Path

# Add LearningHumanoidWalking to path for all processes
sys.path.insert(0, '/home/mehmed-damak/PROJECT09/LearningHumanoidWalking')

# Set PYTHONPATH for Ray workers
os.environ['PYTHONPATH'] = '/home/mehmed-damak/PROJECT09/LearningHumanoidWalking:' + os.environ.get('PYTHONPATH', '')

from rl.algos.ppo import PPO
import argparse
import ray
from functools import partial
from h1_env_wrapper import H1EnvWrapper
#python train_custom_ppo.py ---n-itr 100000 --learning-rate 1.5e-4 --max-traj-len 500 --num-procs 16
def train_with_custom_ppo():
    """Train using LearningHumanoidWalking's PPO implementation"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=Path("./h1_custom_ppo_logs"), type=Path)
    parser.add_argument("--n-itr", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--std-dev", type=float, default=0.223, help="Action noise std")
    parser.add_argument("--learn-std", action="store_true", help="Learn action noise")
    parser.add_argument("--entropy-coeff", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=3, help="Optimization epochs")
    parser.add_argument("--use-gae", type=bool, default=True, help="Use GAE")
    parser.add_argument("--num-procs", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--max-grad-norm", type=float, default=0.05, help="Gradient clipping")
    parser.add_argument("--max-traj-len", type=int, default=1000, help="Max episode length")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency")
    parser.add_argument("--mirror-coeff", type=float, default=0.0, help="Mirror loss coefficient")
    parser.add_argument("--recurrent", action="store_true", help="Use LSTM")
    parser.add_argument("--imitate", type=str, default=None, help="Path to policy to imitate")
    parser.add_argument("--imitate-coeff", type=float, default=0.0, help="Imitation coefficient")
    parser.add_argument("--input-norm-steps", type=int, default=10000, help="Normalization steps")
    parser.add_argument("--continued", type=Path, default=None, help="Continue from checkpoint")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror symmetry")
    parser.add_argument("--yaml", type=str, default=None, help="Path to config file")
    
    args = parser.parse_args()
    
    # Create environment function
    env_fn = partial(H1EnvWrapper)
    
    # Initialize Ray for parallelization
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)
    
    # Create output directory
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    
    # Create custom PPO instance
    print("Creating custom PPO instance...")
    algo = PPO(env_fn, args)
    
    # Train
    print(f"Starting training for {args.n_itr} iterations...")
    algo.train(env_fn, args.n_itr)
    
    print(f"Training complete! Models saved to {args.logdir}")

if __name__ == "__main__":
    train_with_custom_ppo()
