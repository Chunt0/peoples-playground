from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import numpy as np
import gc
from typing import List
import utils
# Segment Anything
def generate_point_mask(image, selected_points, model_type, device, mask_thresh):
    seg_model_dict = utils.get_model_type_dict("segment")
    sam = sam_model_registry[model_type](checkpoint=seg_model_dict[model_type]).to(device)
    sam.mask_threshold = mask_thresh
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
    labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
    transformed_points = predictor.transform.apply_coords_torch(points, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=labels,
        boxes=None,
        multimask_output=False
    )

    masks = masks.cpu().detach().numpy()
    mask_all = np.ones((image.shape[0], image.shape[1], 3))
    for value in masks:
        color_mask = [0,0,0]
        for i in range(3):
            mask_all[value[0] == False, i] = color_mask[i] = color_mask[i]
    masked_image = image /255 * 0.3 + mask_all * 0.7

    sam.cpu()
    points.cpu()
    labels.cpu()

    del sam
    del points
    del labels
    torch.cuda.empty_cache()
    gc.collect()

    return masked_image, mask_all

def organize_box_points(points:List[List[int]]):
    for i in range(0, len(points),2):
        x1 = points[i][0]
        x2 = points[i+1][0]
        y1 = points[i][1]
        y2 = points[i+1][1]
        if x2 < x1:
            points[i][0] = x2
            points[i+1][0] = x1
        if y2 < y1:
            points[i][1] = y2
            points[i+1][1] = y1
    return points


def generate_boxed_mask(image, selected_points, model_type, device, mask_thresh):
    model_dict = utils.get_model_type_dict("segment")
    sam = sam_model_registry[model_type](checkpoint=model_dict[model_type]).to(device)
    sam.mask_threshold = mask_thresh
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    points = organize_box_points([p for p, _ in selected_points])
    boxes = torch.Tensor([points[i]+points[i+1] for i in range(0,len(points),2)]).to(device).unsqueeze(1)
    labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    masks = masks.cpu().detach().numpy()
    mask_all = np.ones((image.shape[0], image.shape[1], 3))
    for value in masks:
        color_mask = [0,0,0]
        for i in range(3):
            mask_all[value[0] == False, i] = color_mask[i] = color_mask[i]
    masked_image = image /255 * 0.3 + mask_all * 0.7


    sam.cpu()
    boxes.cpu()
    labels.cpu()

    del sam
    del points
    del labels
    torch.cuda.empty_cache()
    gc.collect()

    
    return masked_image, mask_all

def run_mask(image, selected_points, model_type, device, boxes, mask_thresh):
    if len(selected_points) != 0 and not boxes:
        return generate_point_mask(image, selected_points, model_type, device, mask_thresh)
    elif len(selected_points) != 0 and (len(selected_points) % 2) == 0 and boxes:
        return generate_boxed_mask(image, selected_points, model_type, device, mask_thresh)
    else:
        print("Something went wrong with your selected points")
        return None, None

# Inpaint
def get_sd_pipe(inpaint_model_type, lora_model_type):  
    sd_model_dict = utils.get_model_type_dict("sd")
    lora_model_dict = utils.get_model_type_dict("lora")
    pipe = AutoPipelineForInpainting.from_pretrained(sd_model_dict[inpaint_model_type], 
                                                     torch_dtype=torch.float16)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    if lora_model_type != "None":
        pipe.load_lora_weights(lora_model_dict[lora_model_type])
        pipe.fuse_lora()
    return pipe


def get_sdxl_pipe(inpaint_model_type, lora_model_type):   
    # TODO: The refiner isn't working. Also should we hard code the VAE? 
    # Getting VAE, hardcoding now but may need to be turned to a global variable
    # For maintenance
    sd_model_dict = utils.get_model_type_dict("sd")
    lora_model_dict = utils.get_model_type_dict("lora")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", 
                                        torch_dtype=torch.float16)    
    
    pipe = AutoPipelineForInpainting.from_pretrained(sd_model_dict[inpaint_model_type], 
                                                     vae=vae,
                                                     torch_dtype=torch.float16)
    '''
    refiner = AutoPipelineForInpainting.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                        text_encoder_2=pipe.text_encoder_2,
                                                        vae=pipe.vae,
                                                        torch_dtype=torch.float16)
    '''
    
    class NoWatermark:
        def apply_watermark(self, image):
            return image
    pipe.watermark = NoWatermark()
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    #refiner.watermark = NoWatermark()
    #refiner.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    if lora_model_type != "None":
        pipe.load_lora_weights(lora_model_dict[lora_model_type])
        pipe.fuse_lora()
        #refiner.load_lora_weights(INPAINT_LORA_MODEL_DICT[lora_model_type])
        #refiner.fuse_lora()
    
    return pipe 

def run_inpaint(
    inpaint_image, 
    segmask_image, 
    pos_prompt, 
    neg_prompt, 
    model_type, 
    lora_type, 
    num_steps,
    num_images, 
    guidance, 
    strength, 
    seed
    ):
    
    num_images = num_images if num_images <= 10 else 10
    
    # Ensure Model Compatability
    if lora_type != "None":
        match = ("xl" in lora_type and "xl" in model_type) or ("xl" not in lora_type and "xl" not in model_type)
        if not match:
            print("### LORA MODEL MUST BE OF SAME TYPE AS INPAINT MODEL ###")
            return None
    
    pipe = get_sdxl_pipe(model_type, lora_type) if "xl" in model_type else get_sd_pipe(model_type, lora_type)

    init_image = load_image(inpaint_image)
    mask_image = load_image(segmask_image)

    width, height = init_image.size
    
    width = width - (width % 8)
    height = height - (height % 8)
    
    generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    pipe.to("cuda")

    images = pipe(prompt=pos_prompt,
                negative_prompt=neg_prompt,
                image=init_image, 
                mask_image=mask_image, 
                generator=generator, 
                height=height, width=width, 
                num_inference_steps=int(num_steps),
                num_images_per_prompt=num_images, 
                guidance_scale=guidance, 
                strength=strength
                ).images
    
    for image in images:
        filename = utils.generate_unique_filename()
        image.save(f"./output/inpaint/{filename}")
        utils.append_to_csv(filename, model_type, lora_type, pos_prompt, neg_prompt, num_steps, guidance, seed, "./output/inpaint/data.csv")
    
    return images[0]
    