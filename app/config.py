import yaml
from pathlib import Path



config = yaml.safe_load(Path('config/application.yml').read_text())