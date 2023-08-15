import os
import sys
import argparse
import tempfile
import requests
from typing import Iterator, Optional

from dotenv import load_dotenv

from examples.dreambooth.train_dreambooth_lora_sdxl_eden import parse_args, main as train_lora

load_dotenv()

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"
os.environ["DIFFUSERS_CACHE"] = "/src/.huggingface/"
os.environ["HF_HOME"] = "/src/.huggingface/"

FIXED_VAE_PATH = "/src/vae_fixed"

#from preprocess_files import load_and_save_masks_and_captions
#from cli_lora_pti import train

from cog import BasePredictor, BaseModel, File, Input, Path

checkpoint_options = {
    "sdxl-v1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}

class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

def download(url, folder, ext):
    try:
        # Making a GET request to fetch the raw image content
        response = requests.get(url)
        response.raise_for_status()
        filename = os.path.basename(urlparse(url).path)
        name_without_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(folder, f'{name_without_ext}.{ext}')

        if not os.path.exists(folder):
            os.makedirs(folder)

        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        image.save(save_path)
        print(f"Downloaded and saved to {save_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def namespace_to_args(namespace):
    args_list = []
    for key, value in vars(namespace).items():
        # Convert the key into a command-line argument
        arg_key = '--' + key.replace('_', '-')
        args_list.append(arg_key)
        # If the value is a boolean flag, only add the key
        if value is not True and value is not False:
            args_list.append(str(value))
    return args_list

class Predictor(BasePredictor):

    def setup(self):
        print("cog:setup")

    def predict(
        self,
        
        checkpoint: str = Input(
            description="Which Stable Diffusion checkpoint to use",
            choices=checkpoint_options.keys(),
            default="sdxl-v1.0"
        ),
        lora_training_urls: str = Input(
            description="Training images for new LORA concept", 
            default=None
        ),
        instance_prompt: str = Input(
            description="a single prompt that describes the concept to be learned", 
            default=None
        ),
        name: str = Input(
            description="Name of new LORA concept",
            default=None
        ),
        resolution: int = Input(
            description="Resolution",
            default=960
        ),
        lr_flip_prob: float = Input(
            description="LR flip prob",
            default=0.5
        ),
        train_batch_size: int = Input(
            description="Batch size",
            default=2
        ),
        gradient_accumulation_steps: int = Input(
            description="Gradient accumulation steps",
            default=2
        ),
        learning_rate: float = Input(
            description="Learning rate for U-Net",
            default=2.5e-4
        ),
        max_train_steps: int = Input(
            description="Max train steps for tuning (U-Net and text encoder)",
            default=500
        ),
        lora_rank: int = Input(
            description="LORA rank",
            default=4
        ),

    ) -> Iterator[CogOutput]:

        print("cog:predict:train_lora")

        # map the checkpoint key to checkpoint path:
        checkpoint_path = checkpoint_options[checkpoint]

        data_dir = Path(tempfile.mkdtemp())
        out_dir  = Path(tempfile.mkdtemp())

        print("train lora", str(data_dir), str(out_dir))

        data_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        
        lora_training_urls = lora_training_urls.split('|')
        for lora_url in lora_training_urls:
            print("download", lora_url)
            download(lora_url, data_dir, '.jpg')

        """
        load_and_save_masks_and_captions(
            files = str(data_dir),
            output_dir = str(data_dir) + "/train",
            caption_text = None,
            target_prompts = "face",
            target_size = 512,
            crop_based_on_salience = True,
            use_face_detection_instead = False,
            temp = 1.0,
            n_length = -1
        )
        """

        ################################################################################################
        print("parsing args..")
        args = argparse.Namespace()
        
        args.pretrained_model_name_or_path = checkpoint_path
        
        #args.pretrained_vae_model_name_or_path = None
        # TODO replace this with relative path + bake new VAE into cog
        args.pretrained_vae_model_name_or_path = FIXED_VAE_PATH

        args.instance_prompt = instance_prompt
        args.instance_data_dir = str(data_dir)
        args.output_dir = str(out_dir)

        args.rank = lora_rank
        args.resolution = resolution
        args.lr_flip_prob = lr_flip_prob
        args.learning_rate = learning_rate
        args.train_batch_size = train_batch_size
        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.dataloader_num_workers = 1
        args.sample_batch_size = 1
        args.max_train_steps = max_train_steps
        #args.train_text_encoder = True  #False by default if not provided
        args.lr_scheduler = "constant_with_warmup"
        args.lr_warmup_steps = 50

        args.checkpointing_steps = args.max_train_steps
        args.checkpoints_total_limit = 1
        args.seed = 0
        args.report_to = "tensorboard"
        args.mixed_precision = "fp16"

        # Regularization (makes training run much slower):
        #args.with_prior_preservation = True  # False by default if not provided
        args.class_data_dir = None
        args.class_prompt = None
        args.prior_loss_weight = 1.0
        args.num_class_images = 0

        print(args)

        # irrelevant, hardcoded args:
        print("Parsing remaining args...")
        manual_args_list = []
        for key, value in vars(args).items():
            manual_args_list.append(f'--{key}')
            manual_args_list.append(str(value))
        args = parse_args(manual_args_list)

        print("Training LORA with args:")
        print(args)

        train_lora(args)

        checkpoint_dir = sorted(os.path.listdir(out_dir))[-1]
        lora_location = os.path.join(str(out_dir), checkpoint_dir, f'pytorch_lora_weights.bin')
    
        yield CogOutput(file=Path(lora_location), name=name, thumbnail=None, attributes=None, isFinal=True, progress=1.0)


if __name__ == "__main__":
    Predictor().run()