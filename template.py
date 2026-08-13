import os 
from pathlib import Path 
import logging 




logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "wasteDetection"

list_of_files = [ 
                 ".github/workflows/.gitkeep",
                "data/.gitkeep",
                f"src/{project_name}/__init__.py",
                f"src/{project_name}/components/__init__.py",
                f"src/{project_name}/components/data_ingestion.py",
                f"src/{project_name}/components/data_validation.py",
                f"src/{project_name}/components/model_trainer.py",
                f"src/{project_name}/components/model_evaluation.py",
                f"src/{project_name}/components/model_pusher.py",
                f"src/{project_name}/constant/__init__.py",
                
                f"src/{project_name}/constant/training_pipeline/__init__.py",
                f"src/{project_name}/constant/application.py",
                f"src/{project_name}/entity/config_entity.py",
                f"src/{project_name}/entity/artifact_entity.py",
                
                f"src/{project_name}/exception/__init__.py",
                f"src/{project_name}/logger/__init__.py",
                f"src/{project_name}/pipeline/__init__.py",
                f"src/{project_name}/pipeline/training_pipeline.py",
                
                f"src/{project_name}/utils/__init__.py",
                f"src/{project_name}/utils/main_utils.py", 
                "template/index.html",
                "app.py",
                "Dockerfile",
                "requirements.txt",
                "setup.py"
                
                

                
]



for filepath in list_of_files: 
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")