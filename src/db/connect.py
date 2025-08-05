from sqlalchemy import create_engine
import yaml

def load_config(path="db_config/user_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_engine():
    cfg = load_config()
    url = f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    return create_engine(url)

