from pathlib import Path
import os
import sys
from dataclasses import dataclass, field
from omegaconf import OmegaConf


this_dir = Path(__file__).parent.resolve()


@dataclass
class FileConfig:
    data_dir: str = "."
    mc_dir: str = "."
    ang_sys_dir: str = "."
    @classmethod
    def load(cls):
        conf = OmegaConf.structured(cls)
        loaded = OmegaConf.load(this_dir / Path("utils.yml"))
        return OmegaConf.merge(conf, loaded)

