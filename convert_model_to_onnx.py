import torch, hydra,logging

from omegaconf.omegaconf import OmegaConf
from model import vgg16
from data import FERDataModule

logger=logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="config",version_base="1.2")
def convert_model(cfg):
    root_dir=hydra.utils.get_original_cwd()
    model_path=f"{root_dir}/models/best_checkpoint.ckpt"
    logger.info(f"Loading model: {model_path}")
    model=vgg16.load_from_checkpoint(model_path)
    
    # load data and prepare
    data=FERDataModule() 
    data.prepare_data()         # load from csv file
    data.setup()                # partition into train+val based on "split"

    # get a batch of input
    input_batch, input_labels=next(iter(data.train_dataloader()))
    
    # export model
    logger.info(f"Converting to ONNX format")
    torch.onnx.export(
        model,
        input_batch,
        f"{root_dir}/models/trained_model.onnx",
        opset_version=15,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0: "batch_size"},
                      "output": {0: "batch_size"}}
    )
    logger.info(f"Model exported sucessfully at {root_dir}/models/trained_model.onnx")

if __name__=="__main__":
    convert_model()