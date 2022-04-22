# In this file, we prepare few-shot data for finetune.py

import utils
import os
from shutil import copy
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_examples_per_label",
                            default=16,
                            type=int,
                            help="number of examples per label.")
    parser.add_argument("--seed",
                            default=100,
                            type=int,
                            help="seed for sampling data.")

    args = parser.parse_args()

    original_data_dir = "/home/s2174572/mlp/mlp1/data/discharge"
    num_examples_per_label = args.num_examples_per_label
    seed = args.seed
    location = 'fewshotdata_for_finetune' # final path will be: /home/s2174572/mlp/mlp1/{location}/few_shot_data

    full_test_set = True # full validation data
    full_val_set = True # full test data

    notes = f"./{location}/notes.txt"
    write_content = f'num_examples_per_label: {num_examples_per_label}\n'
    write_content += f'seed: {seed}\n'
    with open(notes, "w") as writer:
        writer.write(write_content)
    writer.close()
    # ----------------------------------------------------------------------------------------------------------

    # Sample train/val/test datasets
    _ = utils.Group_and_Sample(original_data_dir, num_examples_per_label=num_examples_per_label, seed=seed, location = location)

    if full_val_set == True:
        os.remove(f'/home/s2174572/mlp/mlp1/{location}/few_shot_data/val.csv')
        copy(f'{original_data_dir}/val.csv', f'/home/s2174572/mlp/mlp1/{location}/few_shot_data/val.csv')
    if full_test_set == True:
        os.remove(f'/home/s2174572/mlp/mlp1/{location}/few_shot_data/test.csv')
        copy(f'{original_data_dir}/test.csv', f'/home/s2174572/mlp/mlp1/{location}/few_shot_data/test.csv')

if __name__ == "__main__":
    main()