import gradio as gr
from txt2img import run_txt2img
import utils

def txt2img_tab():
    with gr.Tab("Txt2Img"):
        with gr.Row():
            gr.Markdown("Select Model")
            txt2img_lora_type = gr.Dropdown(choices=list(utils.get_model_type_dict("lora").keys()), value="None", interactive=True, label="Select LoRA Model")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    txt2img_prompt = gr.Textbox(value="", label="Prompt")
                    txt2img_neg_prompt = gr.Textbox(value="", label="Negative Prompt")
                with gr.Row():
                    txt2img_num_steps = gr.Slider(minimum=1.0, maximum=150.0, step=1.0, value=35, label="Number of Steps")
                    txt2img_num_images = gr.Number(value=1, interactive=True, label="Number of Images", precision=0)
                with gr.Row():
                    txt2img_guidance = gr.Slider(minimum=0.01, maximum=50.0, step=0.01, value=7.5, label="Guidance")
                with gr.Row():
                    txt2img_seed = gr.Number(value=-1, label="Seed", interactive=True,precision=0)
                    txt2img_button = gr.Button("txt2img")
            with gr.Column():  
                txt2img_output = gr.Image(type="numpy", interactive=False, height=600)  
    
    
    txt2img_button.click(
        fn=run_txt2img,
        inputs=[txt2img_lora_type,
                txt2img_prompt,
                txt2img_neg_prompt,
                txt2img_num_steps,
                txt2img_num_images,
                txt2img_guidance,
                txt2img_seed],
        outputs = [txt2img_output]
    )
