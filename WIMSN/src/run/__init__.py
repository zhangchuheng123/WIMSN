from .whittle_disc_run import run as whittle_disc_run
from .whittle_cont_run import run as whittle_cont_run
from .iql_run import run as iql_run
from .ippo_run import run as ippo_run

REGISTRY = {}
REGISTRY["whittle_run"] = whittle_disc_run
REGISTRY["whittle_disc_run"] = whittle_disc_run
REGISTRY["whittle_cont_run"] = whittle_cont_run
REGISTRY["iql_run"] = iql_run
REGISTRY["ippo_run"] = ippo_run
