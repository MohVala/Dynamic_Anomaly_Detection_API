import yaml
import uuid
import datetime
from pathlib import Path

class RunContext:
    """
    RunContext holds all run_specifi information:
        Environment (dev/prod)
        config (merged base + env)
        Run ID and timestamps
        flags: use_spark, visaulization, output_dir, api_url
    """

    def __init__(self, env: str = "dev"):
        self.env = env
        self.run_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now()

        # load base config
        base_path = Path("config/base.yaml")
        env_path = Path(f"config/{self.env}.yaml")
        self.config = self._load_yaml(base_path)

        # merge environment specific config
        if env_path.exists():
            env_config = self._load_yaml(env_path)
            self._merge_config(env_config)
        
        # derived flags for easy access:
        self.use_spark = self.config["run"].get("use_spark", False)
        self.visualization = self.config["run"].get("visualization", False)
        self.output_dir = Path(self.config["run"].get("output_dir"))
        self.api_url = self.config["run"].get("api_url")

        # ensure output diroctory exists:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_yaml(self, path: Path) -> dict:
        """load a yaml file safety"""
        with open(path) as f:
            return yaml.safe_load(f)
        

    def _merge_config(self, override_config: dict):
        """
        Recursively merge environment specific config into base config
        """
        for key, value in override_config.items():
            if isinstance(value, dict):
                if key not in self.config:
                    self.config[key] = {}
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def summary(self) -> str:
        """Return a summary string of the run context."""

        return (
            f"Run ID: {self.run_id}\n"
            f"Envirinment: {self.env}\n"
            f"API URL: {self.api_url}\n"
            f"Use Spark: {self.use_spark}\n"
            f"Visualization: {self.visualization}\n"
            f"Output Directory: {self.output_dir}\n"
        )