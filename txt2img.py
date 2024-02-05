import torch
from diffusers import AutoencoderKL, DiffusionPipeline
import utils

def load_sdxl(lora_model_type):
    # SDXL has a watermark, this is a hack to remove it
    lora_model_dict = utils.get_model_type_dict("lora")
    class NoWatermark:
        def apply_watermark(self, img):
            return img 
            
    model_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
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

    return pipe
    
def generate_sdxl(model, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed):
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
    for image in images:
        filename = utils.generate_unique_filename()
        image.save(f"./output/txt2img/{filename}")

    return images[0]
    

def run_txt2img(lora_type,
                pos_prompt, 
                neg_prompt, 
                num_steps,
                num_images,
                guidance,
                seed):
    num_images = num_images if num_images <= 10 else 10
    model = load_sdxl(lora_type)
    return generate_sdxl(model, pos_prompt, neg_prompt, num_steps, num_images, guidance, seed)