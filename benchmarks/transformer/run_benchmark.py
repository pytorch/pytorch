#!/usr/bin/env python3
"""
Wrapper script to run score_mod.py with YAML configuration files.
Converts YAML configs to CLI arguments for score_mod.py.
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def config_to_args(config):
    """Convert YAML config to CLI arguments for score_mod.py"""
    args = []
    
    # Core parameters
    if config.get('dynamic'):
        args.append('--dynamic')
    
    if config.get('calculate_bwd'):
        args.append('--calculate-bwd')
    
    if config.get('dtype'):
        args.extend(['-dtype', config['dtype']])
    
    # Shape parameters - directly use arrays from YAML
    if config.get('b'):
        args.extend(['-b'] + config['b'])
    
    if config.get('nh'):
        args.extend(['-nh'] + config['nh'])
    
    if config.get('s'):
        args.extend(['-s'] + config['s'])
    
    if config.get('d'):
        args.extend(['-d'] + config['d'])
    
    if config.get('mods'):
        args.extend(['-mods'] + config['mods'])
    
    # Backend and optimization - directly use arrays
    if config.get('backend'):
        args.extend(['--backend'] + config['backend'])
    
    if config.get('max_autotune'):
        args.append('--max-autotune')
    
    # Decoding and cache settings
    if config.get('decoding'):
        args.append('--decoding')
    
    if config.get('kv_size'):
        args.extend(['--kv-size'] + config['kv_size'])
    
    # Metrics and output
    if config.get('throughput'):
        args.append('--throughput')
    
    # Always ensure save_path is set for CSV output
    save_path = config.get('save_path')
    if save_path:
        args.extend(['--save-path', save_path])
    else:
        # Generate default filename if not specified
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"benchmark_results_{timestamp}.csv"
        args.extend(['--save-path', default_path])
    
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run score_mod.py with YAML configuration files"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Print the command that would be executed without running it"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Convert to CLI arguments
    cli_args = config_to_args(config)
    
    # Build command - convert all args to strings for subprocess
    cmd = ['python', 'score_mod.py'] + [str(arg) for arg in cli_args]
    
    if args.dry_run:
        print("Command that would be executed:")
        print(' '.join(cmd))
        print(f"\nOutput file: {config.get('save_path', 'benchmark_results_TIMESTAMP.csv')}")
        return
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    print(f"Results will be saved to: {config.get('save_path', 'benchmark_results_TIMESTAMP.csv')}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print("=" * 80)
            print("Benchmark completed successfully!")
        else:
            print("=" * 80)
            print(f"Benchmark failed with exit code: {result.returncode}")
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
