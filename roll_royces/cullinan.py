# Cullinan running script - 🐳 Build for running in Docker

import os
import random
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from modules.alias import get_by_aliases
from modules.utils.preprocessing import Preprocessing
from modules.utils.missing import CreateMissingDataFrame
from modules import Trainer
from modules.models import *
from modules.utils.callbacks import SavePlot, Combined
from modules.utils.generator import WindowGenerator
from modules.utils.cache import Cache


matplotlib.use('Agg')
SAVE_DIR = "/result"


def __get_random_seed(seed_bounder: int = 10000) -> int:
    """
    Get the random seed.
    """
    return random.randint(0, seed_bounder)


def __parse_selected_models(models: str) -> list:
    """
    Parse selected models.
    """
    return map(str.strip, models.split(','))


def cullinan_attack(params: dict):
    dataset = params.get('dataset') or 'PhuLien'
    missing_percentage = params.get('missing_percentage') or 6
    missing_gaps = params.get('missing_gaps') or 1
    seed = params.get('seed') or __get_random_seed()
    mode = params.get('mode') or 'Random'
    window_size = params.get('window_size') or missing_percentage
    batch_size = params.get('batch_size') or 1
    models = __parse_selected_models(params.get('models') or 'lr')
    combination_mode = params.get('combination_mode') or 'mean'

    print('⚙️ Configuration')
    print(f'Dataset: {dataset}')
    print(f'Missing percentage: {missing_percentage}')
    print(f'Missing gaps: {missing_gaps}')
    print(f'Seed: {seed}')
    print(f'Mode: {mode}')
    print(f'Window size: {window_size}')
    print(f'Batch size: {batch_size}')
    print(f'Models: {models}')
    print(f'Combination mode: {combination_mode}')
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.style.use('ggplot')

    # Get models
    models = get_by_aliases(models)

    # Read CSV
    df = pd.read_csv(f'/data/{dataset}.csv')
    df = df[[df.columns[-1]]]

    # Preprocessing data
    preprocessing = Preprocessing()
    df = preprocessing.flow(df)

    # Create missing data
    creator = CreateMissingDataFrame(df, missing_percentage, missing_gaps, split_mode=mode,
                                     seed=seed, is_constant_missing=True, safe_random_window=window_size)
    creator.plot(save_path=f'{SAVE_DIR}/missing_data.png')

    # Training
    print('🚀 Training')
    trainer = Trainer(model=models)
    combined_callback = Combined(n_models=len(
        models), combination_mode=combination_mode, df=creator, save_directory=SAVE_DIR)
    sp = SavePlot(n_models=len(models), save_directory=SAVE_DIR)
    cache = Cache(live_cache=True)

    for train_df, test_df in creator:
        train_gen = WindowGenerator(train_df, window_size, batch_size)
        test_gen = WindowGenerator(test_df, window_size, batch_size)

        trainer.train(train_gen, test_gen, callbacks=[
                      sp, combined_callback], cache=cache)
        trainer.reset()

    # Save metrics
    combined_callback.metrics.metrics.to_csv(
        f'{SAVE_DIR}/metrics.csv', index=False)

    print('🎉 Done!\n', str(combined_callback.metrics.metrics))


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Cullinan running script')

    # Add arguments
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--missing_percentage', type=int,
                        help='Missing percentage')
    parser.add_argument('--missing_gaps', type=int, help='Missing gaps')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--mode', type=str,
                        help='Missing mode', choices=['Random', 'Linear'])
    parser.add_argument('--window_size', type=int,
                        help='Window size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--models', type=str,
                        help='Models')
    parser.add_argument('--combination_mode', type=str, help='Combination mode',
                        choices=['mean', 'data_per', 'similarity', 'meow'])

    cullinan_attack(vars(parser.parse_args()))
