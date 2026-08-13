import sys 
import os 
import logging 

logging_stream = logging.StreamHandler(sys.stdout) 

log_dir = "logs" 
log_filepath = os.path.join(log_dir, "running_logs.log") 
os.makedirs(log_dir, exist_ok=True) 

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging_stream, logging.FileHandler(log_filepath, mode="a")],
    format="[%(asctime)s]: %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
