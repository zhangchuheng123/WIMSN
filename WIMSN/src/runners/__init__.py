REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .evaluate_runner import EvaluateRunner
REGISTRY["evaluate"] = EvaluateRunner

from .whittle_cont_runner import WhittleContinuousRunner
REGISTRY["whittle_cont"] = WhittleContinuousRunner