"""
yaml 文件
k: v 即可
"""

import yaml
from backend.utils.path_tool import get_abs_path

def load_model_config(config_path: str=get_abs_path("config/model.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)    # FullLoader 
    
def load_chroma_config(config_path: str=get_abs_path("config/chroma.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)    # FullLoader 

def load_prompts_config(config_path: str=get_abs_path("config/prompts.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)    # FullLoader 

def load_agent_config(config_path: str=get_abs_path("config/agent.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)    # FullLoader

def load_ingestion_config(config_path: str=get_abs_path("config/ingestion.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def load_short_memory_config(config_path: str=get_abs_path("config/short_memory.yaml"), encoding: str='utf-8') -> dict:
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

model_conf = load_model_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
ingestion_conf = load_ingestion_config()
short_memory_conf = load_short_memory_config()

if __name__ == '__main__':
    print(model_conf['chat_model_name'])