from PIL import Image
import torch
import os
from tqdm import tqdm
import gc
import sys

import matplotlib.pyplot as plt
import ollama
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
import re

import inspect
# Semantic SAM
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from semantic_sam.BaseModel import BaseModel
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam import build_model
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.modeling import build_model as build_model_seem

from task_adapter.seem.tasks import inference_seem_pano

prompt_1 = """
    USER: <image>
    Each number tags represent a semantic region defined by the masks. 
    The black block with number in the middle in first color are the tags for the regions.
    Assign a label to each number tag based on the region.
    If the region is not relevant, just ignore it.

    Follow this **exact** format: {number:label}. 

    Example: An image with 3 regions, where the first region is a cat with number tag 1, 
    the second region is a dog with number tag 2, and the third region is a tree with number tag 3.
    Output: {1:cat, 2:dog, 3:tree}

    No need to include any extra information in your response.
    ASSISTANT:
"""

prompt_2 = """
        USER: <image>
        I have labeled a bright numeric ID at the center for each visual object in the image.
        Please enumerate their names.Assign a label to each number tag based on the region.
        Follow this **exact** format: {number:label}.
        No need to include any extra information in your response.
        ASSISTANT:
"""

level = [2]
alpha = 0.1
anno_mode = ["Mark"]
label_mode = "1"
text, text_part, text_thresh = "", "", "0.0"
text_size, hole_scale, island_scale = 640, 100, 100
semantic = True

semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./checkpoints/" + "swinl_only_sam_many2many.pth"
seem_ckpt = "./checkpoints/" + "seem_focall_v1.pt"

model_path = "zzxslp/som-llava-v1.5-13b-hf"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  
#     bnb_4bit_use_double_quant=True, 
#     bnb_4bit_quant_type="nf4",  
#     bnb_4bit_compute_dtype=torch.float16,
# )

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True)



BASE_OUTPUT = "./outputs"

def load_semsam_model():
    opt = load_opt_from_config_file(semsam_cfg)
    opt['device'] = "cuda:0"
    model = BaseModel(opt, build_model(opt)).from_pretrained(semsam_ckpt).eval().cuda()
    return model

def load_seem_model():
    opt = load_opt_from_config_file(seem_cfg)
    opt['device'] = "cuda:0"
    model = BaseModel_Seem(opt, build_model_seem(opt)).from_pretrained(seem_ckpt).eval().cuda()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
    return model

def load_llava_hf_model():
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def get_mask_label(image, vision_model, name, prompt, prompt_index, ollama_):

    if vision_model == "semsam":
        model_semsam = load_semsam_model()
    elif vision_model == "seem":
        model_seem = load_seem_model()

    with torch.amp.autocast(device_type="cuda"):
        if vision_model == "semsam":
            output, mask, masks, result_image = inference_semsam_m2m_auto(
                model_semsam,
                image,
                level,
                text,
                text_part,
                text_thresh,
                text_size,
                hole_scale,
                island_scale,
                semantic,
                label_mode=label_mode,
                alpha=alpha,
                anno_mode=anno_mode,
            )
        elif vision_model == "seem":
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

            output, mask, masks, result_image = inference_seem_pano(model_seem, image, text_size, label_mode, alpha, anno_mode)

    image_path = os.path.join(BASE_OUTPUT, vision_model)

    path1 = os.path.join(image_path, f"{name}_result.jpg")
    plt.imshow(result_image)
    plt.savefig(path1)

    path2 = os.path.join(image_path, f"{name}_output.jpg")
    plt.imshow(output)
    plt.savefig(path2)

    if vision_model == "semsam":
        del model_semsam
    elif vision_model == "seem":
        del model_seem

    torch.cuda.empty_cache()
    gc.collect()

    image = Image.open(path1).convert("RGB")

    if prompt_index == "1":
        prompt_path = f"prompt1/{name}.txt"
    else:
        prompt_path = f"prompt2/{name}.txt"

    if not ollama_:
        llava_model, llava_processor = load_llava_hf_model()
        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to("cuda")

        generate_ids = llava_model.generate(**inputs, max_new_tokens=512,
                                            eos_token_id=llava_model.config.eos_token_id)
        output = llava_processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Save output
        os.makedirs(os.path.join(image_path, "huggingface", "prompt1"), exist_ok=True)
        os.makedirs(os.path.join(image_path, "huggingface", "prompt2"), exist_ok=True)

        text_path = os.path.join(image_path, "huggingface", prompt_path)
        with open(text_path, "w") as f:
            f.write(output)

        del llava_model, llava_processor
        torch.cuda.empty_cache()
        gc.collect()
    else:
        # Llava 1.6
        res = ollama.chat(
            model="llava:13b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [path1],
                }
            ]
        )
        output = res['message']['content']

        os.makedirs(os.path.join(image_path, "ollama", "prompt1"), exist_ok=True)
        os.makedirs(os.path.join(image_path, "ollama", "prompt2"), exist_ok=True)

        text_path = os.path.join(image_path, "ollama", prompt_path)
        with open(text_path, "w") as f:
            f.write(output)

def main():
    BASE = "./images"

    for name in tqdm(os.listdir(BASE)):
        image_path = os.path.join(BASE, name)
        image = Image.open(image_path)
    
        for vision_model in ["semsam", "seem"]:
            for i, prompt in enumerate([prompt_1, prompt_2]):
                for ollama_ in [True, False]:
                    get_mask_label(image, vision_model, name, prompt, str(i + 1), ollama_)

                    torch.cuda.empty_cache()
                    gc.collect()

def data_preparation(vision_model):
    BASE_IMAGE = "../downloads"
    BASE_OUTPUT = "./train"

    os.makedirs(BASE_OUTPUT, exist_ok=True)

    prompt = r"""
        USER: <image>
        I have labeled a bright numeric ID at the center for each visual object in the image.
        Assign a label to each number tag based on the region.
        Follow this **exact** format: {number:label}.
        Do not include any extra information in your response apart from label.
        Please assign a concise name for them.
        ASSISTANT:
    """

    if vision_model == "semsam":
        model_semsam = load_semsam_model()
    elif vision_model == "seem":
        model_seem = load_seem_model()

    llava_model, llava_processor = load_llava_hf_model()

    def inference_image(image):
        with torch.amp.autocast(device_type="cuda"):
            if vision_model == "semsam":
                output, mask, masks, result_image = inference_semsam_m2m_auto(
                    model_semsam,
                    image,
                    level,
                    text,
                    text_part,
                    text_thresh,
                    text_size,
                    hole_scale,
                    island_scale,
                    semantic,
                    label_mode=label_mode,
                    alpha=alpha,
                    anno_mode=anno_mode,
                )
            elif vision_model == "seem":
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

                output, mask, masks, result_image = inference_seem_pano(model_seem, image, text_size, label_mode, alpha, anno_mode)

        return output, result_image

    def inference_text(image):
        print(image)
        print(prompt)
        llava_processor.patch_size = 14
        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        generate_ids = llava_model.generate(**inputs, max_new_tokens=512,
                                            eos_token_id=llava_model.config.eos_token_id)
        output = llava_processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output

    for i, folder in tqdm(enumerate(os.listdir(BASE_IMAGE))):
        folder_path = os.path.join(BASE_IMAGE, folder, "images", "scene_cam_00_final_preview")
        path_1 = os.path.join(folder_path, "frame.0000.color.jpg")
        path_2 = os.path.join(folder_path, "frame.0001.color.jpg")
        
        scene = f"scene_{i}"

        image_1 = Image.open(path_1).convert("RGB")
        image_2 = Image.open(path_2).convert("RGB")

        output_1, result_image_1 = inference_image(image_1)
        output_2, result_image_2 = inference_image(image_2)

        OUTPUT_PATH = os.path.join(BASE_OUTPUT, scene)
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        path1 = os.path.join(OUTPUT_PATH, f"{scene}_image1_result.jpg")
        plt.imshow(result_image_1)
        plt.savefig(path1)

        path2 = os.path.join(OUTPUT_PATH, f"{scene}_image2_result.jpg")
        plt.imshow(result_image_2)
        plt.savefig(path2)

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        output_1 = inference_text(img1)
        output_2 = inference_text(img2)

        output_1 = output_1.split("ASSISTANT:")[-1]
        output_2 = output_2.split("ASSISTANT:")[-1]

        def filter_output(text):
            lines = text.strip().split('\n')
            filtered_lines = [line for line in lines if re.match(r'^\s*\d+\.', line)]
            result = '\n'.join(filtered_lines)
            return result
        
        output_1 = filter_output(output_1)
        output_2 = filter_output(output_2)

        with open(OUTPUT_PATH + f"{scene}_image1.txt", "w") as f:
            f.write(output_1)

        with open(OUTPUT_PATH + f"{scene}_image2.txt", "w") as f:
            f.write(output_2)

if __name__ == "__main__":
    data_preparation("semsam")
    #main()
   