from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, num_episodes, policy):
        pass

    def callback(self, logging_info: dict):
        pass

    def callback_verbose(self, wandb_logger):
        pass


class HeadlessEvaluator(Evaluator):
    """No display / no sim: skip sim evaluation, only validation loss. Use for RLBench on headless servers."""

    def __init__(self, *args, **kwargs):
        """
        Accept arbitrary arguments so it can be instantiated via Hydra
        with any config fields (e.g. task_name) without error.
        """
        super().__init__()

    def evaluate(self, num_episodes, policy):
        return -1.0, -1.0  # so best_success is never updated by sim

    def callback(self, logging_info: dict):
        pass
