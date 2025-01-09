import os

import numpy as np
import argparse
import torch
import random

from models.p2p_editor import P2PEditor

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./images/farm.png") # the editing category that needed to run
    parser.add_argument('--original_prompt', type=str, default="a farm") # the editing category that needed to run
    parser.add_argument('--editing_prompt', type=str, default="an lighthouse on farm") # the editing category that needed to run
    parser.add_argument('--blended_word', type=str, default="farm farm") # the editing category that needed to run
    parser.add_argument('--output_path', nargs = '+',type=str, default=["farm_directinversion+p2p_guidance_75_75_3.jpg"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["directinversion+p2p_guidance_75_75"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    output_path=args.output_path
    edit_method_list=args.edit_method_list

    p2p_editor=P2PEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )
    
    original_prompt = args.original_prompt
    editing_prompt = args.editing_prompt
    image_path = args.image_path
    blended_word = args.blended_word.split(" ") if args.blended_word != "" else []

    for edit_method_i in range(len(edit_method_list)):
        edit_method=edit_method_list[edit_method_i]
        present_image_save_path=output_path[edit_method_i]
    
        print(f"editing image [{image_path}] with [{edit_method}]")
        setup_seed()
        torch.cuda.empty_cache()
        edited_image = p2p_editor(edit_method,
                                    image_path=image_path,
                                prompt_src=original_prompt,
                                prompt_tar=editing_prompt,
                                guidance_scale=7.5,
                                cross_replace_steps=0.4,
                                self_replace_steps=0.6,
                                blend_word=(((blended_word[0], ),
                                            (blended_word[1], ))) if len(blended_word) else None,
                                eq_params={
                                    "words": (blended_word[1], ),
                                    "values": (2, )
                                } if len(blended_word) else None,
                                proximal="l0",
                                quantile=0.75,
                                use_inversion_guidance=True,
                                recon_lr=1,
                                recon_t=400,
                                )
        
        edited_image.save(present_image_save_path)
        
        print(f"finish")
            
