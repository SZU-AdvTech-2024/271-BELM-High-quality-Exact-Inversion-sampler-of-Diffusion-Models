import sys

import torch
import os
import json
import argparse
import random
from tqdm import tqdm
sys.path.append(os.getcwd())
from samplers import test_sd15, BELM, BDIA, edict, DDIM
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler
from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
from samplers.utils import PipelineLike


def load_coco_descriptions(coco_annotations_path):
    """
    从 COCO 验证集注释文件中读取所有图片的描述（captions）。
    返回一个列表，包含所有的描述。
    """
    with open(coco_annotations_path, 'r') as f:
        annotations = json.load(f)
    captions = [ann['caption'] for ann in annotations['annotations']]
    return captions


def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO2014 on chongqing machine.")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--guidance', type=float, default=5.5)
    parser.add_argument('--sampler_type', type=str, default='belm', choices=['lag', 'ddim', 'bdia', 'edict', 'belm'])
    parser.add_argument('--save_dir', type=str, default='./test_results/sd_coco_images')
    parser.add_argument('--model_id', type=str, default='xxx/stable-diffusion-v1-5')
    parser.add_argument('--coco_annotations_path', type=str, default="./data/coco2014/captions_val2014.json",
                        help='Path to COCO2014 val annotations (captions_val2014.json)')
    parser.add_argument('--total_images', type=int, default=10000, help='Total number of images to generate')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    args = parser.parse_args()

    # 加载 COCO 描述
    captions = load_coco_descriptions(args.coco_annotations_path)
    print(f"Loaded {len(captions)} captions from COCO annotations.")

    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    model_id = args.model_id
    dtype = torch.float32
    # 初始化模型和调度器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False,
                         timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device=device, vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                           unet=sd.unet, scheduler=sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('model loaded')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 随机选择描述并生成图片
    batch_size = min(args.batch_size, args.total_images)  # 动态调整 batch size
    total_batches = (args.total_images + batch_size - 1) // batch_size

    print(f"Generating {args.total_images} images in {total_batches} batches of size {batch_size}.")
    for batch_idx in tqdm(range(total_batches), desc="Generating images"):
        current_batch_size = min(batch_size, args.total_images - batch_idx * batch_size)
        prompts = random.sample(captions, current_batch_size)
        negative_prompt = ["" for _ in range(current_batch_size)]
        # intermediate to latent
        sd_params = {'prompt': prompts, 'negative_prompt': negative_prompt, 'seed': 38018716,
                     'guidance_scale': guidance_scale,
                     'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}

        if sampler_type in ['ddim']:
            result_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,
                                                        freeze_step=0)
        elif sampler_type in ['edict']:
            result_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, x_intermediate=None,
                                                            y_intermediate=None, p=args.edict_p, freeze_step=0)
        elif sampler_type in ['bdia']:
            result_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,
                                                        intermediate_second=None, gamma=args.bdia_gamma, freeze_step=0)
        elif sampler_type in ['lag', 'belm']:
            result_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                        intermediate=None,
                                                        intermediate_second=None, freeze_step=0)

        images = test_sd15.to_pil(latents=result_latent, sd_pipe=sd_pipe)

        # 保存图片
        for i, image in enumerate(images):
            image.save(os.path.join(args.save_dir, f"generated_{batch_idx * batch_size + i + 1}.png"))

    print(f"Image generation completed. Images saved to {args.save_dir}.")


if __name__ == "__main__":
    main()