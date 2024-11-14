import multiprocessing

import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str = r"C:\Users\28024\Desktop\MindNLP\论文解读\human-eval\data\example_samples.jsonl",
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 5.0,
    problem_file: str = r"C:\Users\28024\Desktop\MindNLP\论文解读\human-eval\data\human-eval-v2-20210705.jsonl",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)


def main():
    fire.Fire(entry_point)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

