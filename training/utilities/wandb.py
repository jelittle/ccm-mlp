try:
    import wandb
except ImportError:  # optional: training runs offline without it
    wandb = None
import os
from datetime import datetime


def setup_wandb(config,device):
    # Extract values from config
    data_config = config['data']
    model_config = config['model']

    now = datetime.now()
    timestamp = now.strftime("%m_%d_%H_%M_%S")
    run_name = f"{data_config['camera_name']}_{model_config['type']}_{data_config['split']}_{timestamp}"

    wandb.init(project=config["wandb"].get("project", "CST-Calibration"), entity=config["wandb"].get("entity") or None, name=run_name, mode='disabled' if config['wandb']['disabled'] else 'online')
    #add run information to wandb
    wandb.config.update({
        "epochs": model_config['epochs'],
        "batch_size": data_config['batch_size'],
        "learning_rate": model_config['lr'],
        "device": str(device),
        "model_type": model_config['type'],
        "chart_type": data_config['chart_type'],
    })
    return run_name