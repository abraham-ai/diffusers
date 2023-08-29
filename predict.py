import os
import sys
import shutil
import argparse
import tempfile
import requests
from typing import Iterator, Optional
from dotenv import load_dotenv
from PIL import Image

DEBUG_MODE = False
#DEBUG_MODE = True

from io_utils import make_validation_img_grid, download_and_prep_training_data

from examples.dreambooth.train_dreambooth_lora_sdxl_eden import parse_args, main as train_lora
#from examples.dreambooth.train_dreambooth_lora_sdxl import parse_args, main as train_lora

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
    files: list[Path]
    name: Optional[str] = None
    thumbnails: Optional[list[Path]] = []
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

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

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
            description="Training images for new LORA concept (can be images or a .zip file of images)", 
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
        lr_flip_p: float = Input(
            description="Left-right flipping probability for data augmentation",
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
            default=3e-4
        ),
        max_train_steps: int = Input(
            description="Max train steps for tuning (U-Net and text encoder)",
            default=400
        ),
        lora_rank: int = Input(
            description="LORA rank",
            default=4
        ),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:

        print("cog:predict:train_lora")

        # map the checkpoint key to checkpoint path:
        checkpoint_path = checkpoint_options[checkpoint]

        data_dir = Path(tempfile.mkdtemp())
        out_dir  = Path(tempfile.mkdtemp())

        # Local test to see images:
        out_dir = Path("test_lora_out2")
        os.makedirs(out_dir, exist_ok=True)

        print("train lora", str(data_dir), str(out_dir))

        data_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        download_and_prep_training_data(lora_training_urls, data_dir)

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
        args.lr_flip_p = lr_flip_p
        args.learning_rate = learning_rate
        args.train_batch_size = train_batch_size
        args.gradient_accumulation_steps = gradient_accumulation_steps
        args.dataloader_num_workers = 1
        args.sample_batch_size = 1
        args.max_train_steps = max_train_steps
        #args.train_text_encoder = True  #False by default if not provided
        args.lr_scheduler = "constant_with_warmup"
        args.lr_warmup_steps = 25

        args.checkpointing_steps = args.max_train_steps
        args.num_validation_images = 4
        args.validation_prompt = instance_prompt
        args.validation_epochs = 1000
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
        
        checkpoint_dir = sorted([f for f in os.listdir(args.output_dir) if 'checkpoint' in f])[-1]
        lora_location = os.path.join(args.output_dir, checkpoint_dir, 'pytorch_lora_weights.bin')

        attributes = {
            "instance_prompt": instance_prompt,
        }

        print("LORA trainig done, returning cog output")
        print("lora_location: ", lora_location)
        print("Trigger prompt: ", instance_prompt)

        # save all the args into a dict:
        args_dict = {}
        for key, value in vars(args).items():
            args_dict[key] = str(value)

        # add the trigger prompt:
        args_dict['instance_prompt'] = instance_prompt

        # save the args dict as a json file:
        import json
        with open(os.path.join(out_dir, 'args.json'), 'w') as f:
            json.dump(args_dict, f)

        # remove all the subfolders in out_dir that start with checkpoint-:
        for f in os.listdir(out_dir):
            if f.startswith('checkpoint-'):
                shutil.rmtree(os.path.join(out_dir, f))

        # Zip the output folder:
        shutil.make_archive(os.path.join(os.getcwd(), "lora"), 'zip', out_dir)
        return_filepath = os.path.join(os.getcwd(), "lora.zip")

        print(f"Saved LORA weights and settings to {return_filepath}")

        validation_grid_img_path = make_validation_img_grid(out_dir)
        
        if DEBUG_MODE:
            yield Path(return_filepath)
        else:
            yield CogOutput(files=[Path(return_filepath)], name=name, thumbnails=[Path(validation_grid_img_path)], attributes=attributes, isFinal=True, progress=1.0)


