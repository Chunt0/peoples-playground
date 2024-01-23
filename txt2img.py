import torch
import gc
from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline
import utils
from PIL import Image

def load_sd(model_type, lora_model_type, vae_model_type):
    sd_model_dict = utils.get_model_type_dict("sd")
    lora_model_dict = utils.get_model_type_dict("lora")
    model_checkpoint = sd_model_dict[model_type]
    lora_checkpoint = lora_model_dict[lora_model_type]

    pipe = StableDiffusionPipeline.from_pretrained(model_checkpoint, torch_dtype=torch.float16)
    if lora_model_type != "None":
        pipe.load_lora_weights(lora_checkpoint)
        pipe.fuse_lora()
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.to("cuda")
    
    return pipe, None

def load_sdxl(model_type, lora_model_type, vae_model_type):
    # SDXL has a watermark, this is a hack to remove it
    sd_model_dict = utils.get_model_type_dict("sd")
    lora_model_dict = utils.get_model_type_dict("lora")
    class NoWatermark:
        def apply_watermark(self, img):
            return img 
            
    model_checkpoint = sd_model_dict[model_type]
    lora_checkpoint = lora_model_dict[lora_model_type]
    
    # Get VAE for SDXL this is hardcoded
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # Load Tuned Base model and refiner
    pipe = DiffusionPipeline.from_pretrained(
        model_checkpoint, 
        torch_dtype=torch.float16,
        vae=vae,
        )
    if lora_model_type != "None":
        pipe.load_lora_weights(lora_checkpoint)
        pipe.fuse_lora()
    pipe.watermark = NoWatermark()
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    pipe.to("cuda")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )    
    refiner.watermark = NoWatermark()
    refiner.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    refiner.to("cuda") 

    return pipe, refiner

def load_model(model_type, lora_model_type, vae_model_type):
    if lora_model_type != "None":
        match = ("xl" in lora_model_type and "xl" in model_type) or ("xl" not in lora_model_type and "xl" not in model_type)
        if not match:
            print("### LORA MODEL MUST BE OF SAME TYPE AS INPAINT MODEL ###")
            return None
    if "xl" in model_type:
        return load_sdxl(model_type, lora_model_type, vae_model_type)
    else:
        return load_sd(model_type, lora_model_type, vae_model_type)

    
def generate_sdxl(model, refiner, model_type, lora_type, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed):
    generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    images = model(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=num_steps,
        num_images_per_prompt=num_images,
        guidance_scale=guidance,
        #output='latent',
        generator=generator,
    ).images
    '''
    images = refiner(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=num_steps,
        num_images_per_prompt=num_images,
        guidance_scale=guidance,
        image=image,
        generator=generator
    ).images
    '''    
    for image in images:
        filename = utils.generate_unique_filename()
        image.save(f"./output/txt2img/{filename}")
        utils.append_to_csv(filename, model_type, lora_type, pos_prompt, neg_prompt, num_steps, guidance, seed, "./output/txt2img/data.csv")

    return images[0]
    

def generate_sd(model, model_type, lora_type, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed):
    generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)
    images = model(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=num_steps,
        num_images_per_prompt=num_images,
        guidance_scale=guidance,
        generator=generator
    ).images
    
    for image in images:
        filename = utils.generate_unique_filename()
        image.save(f"./output/inpaint/{filename}")
        utils.append_to_csv(filename, model_type, lora_type, pos_prompt, neg_prompt, num_steps, guidance, seed, "./output/txt2img/data.csv")
    
    return images[0]
    

def run_txt2img(model,
                refiner,
                model_type,
                lora_type,
                pos_prompt, 
                neg_prompt, 
                num_steps,
                num_images,
                guidance,
                seed):
    num_images = num_images if num_images <= 10 else 10
    
    
    if "xl" in model_type:
        model, refiner = load_sdxl(model_type, lora_type, None)
        return generate_sdxl(model, model_type, lora_type, refiner, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed)
    
    else:
        model, refiner = load_sd(model_type, lora_type, None)
        return generate_sd(model, model_type, lora_type, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed)