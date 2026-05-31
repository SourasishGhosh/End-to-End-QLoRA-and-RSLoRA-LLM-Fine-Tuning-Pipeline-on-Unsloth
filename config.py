import os
import os.path as op

class BaseConfig(object):
    BASEDIR = op.abspath(op.dirname(__file__))
    ROOT_DIR = BASEDIR

    MODEL_NAME = "unsloth/gemma-2-2b-it"

    LORA_R = 16
    LORA_ALPHA = 16 
    LORA_DROPOUT = 0.0 
    
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    EPOCHS = 1
    BATCH_SIZE = 1 
    LEARNING_RATE = 2e-4 
    MAX_OUTPUT_TOKEN = 50
    TOP_K = 64
    TOP_P = 0.95
    TEMPERATURE = 1.0

    MAX_LENGTH = 512

    OUTPUT_DIR = "./gemma_outputs"

Config = BaseConfig()