import gradio as gr
from segment import run_inpaint, run_mask
import utils

def segment_tab(device):
    with gr.Tab(label="Segment Image"):
        with gr.Row():
            gr.Markdown('# Segment Anything')
            # Select the model to use and the device to use
            seg_model_type = gr.Dropdown(choices=list(utils.get_model_type_dict("segment").keys()), value="None", interactive=True, label="Select SAM Model")
            mask_thresh = gr.Slider(value=0.0, minimum=-10.0, maximum=10.0, step=0.01, interactive=True,
                                    label="Mask Threshold",
                                    info="Determine how sensitive the predictive model is, lower makes it less sensitive.")
        with gr.Tab(label="Image"):                                  
            with gr.Row(equal_height=True):
                with gr.Column():
                    seg_original_image = gr.State(value=None)
                    seg_input_image = gr.Image(type="numpy", height=600)
                    with gr.Column():
                        selected_point = gr.State()
                        with gr.Row():
                            boxes = gr.Checkbox(label="Bounding Boxes", interactive=True)
                            seg_undo_button = gr.Button('Undo point')
                        point_type = gr.Radio(choices=['foreground_point', 'background_point'], value='foreground_point', interactive=True, label='point labels')
                    # run button
                    seg_button = gr.Button("Segment Anything")
                # show the image with mask
                with gr.Tab(label='Image+Mask'):
                    seg_output_image = gr.Image(type='numpy', interactive=False, height=600)
                # show only mask
                with gr.Tab(label='Mask'):
                    seg_output_mask = gr.Image(type='numpy', interactive=False, height=600)
    with gr.Tab(label="Inpaint"):
        with gr.Row():
            gr.Markdown("# Inpaint with Seg Mask")
            inpaint_model_type = gr.Dropdown(choices=list(utils.get_model_type_dict("sd").keys()), value="None", interactive=True, label="Select Inpainting Model")
            inpaint_lora_type = gr.Dropdown(choices=list(utils.get_model_type_dict("lora").keys()), value="None", interactive=True, label="Select LoRA Model")
            inpaint_vae_model_type = gr.Dropdown(choices=list(utils.get_model_type_dict("vae").keys()), value="None", interactive=True, label="Select VAE" )
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row():
                    inpaint_image = gr.Image(value="./examples/inpaint_this.jpg", type="pil", interactive=True, label="Original Image", height=300)
                    inpaint_mask_image = gr.Image(value="./examples/with_this_mask.jpg", type="pil", interactive=True, label="Image Mask", height=300)
                with gr.Row():
                    inpaint_pos_prompt = gr.Textbox(value="", label="Positive Prompt", interactive=True)
                    inpaint_neg_promt = gr.Textbox(value="", label="Negative Prompt", interactive=True)
                with gr.Row():
                    inpaint_num_steps = gr.Slider(minimum=1.0, maximum=150.0, step=1.0, value=35.0, label="Number of Steps")
                    inpaint_num_images = gr.Number(value=1, interactive=True, label="Number of Images", precision=0)
                with gr.Row():
                    inpaint_guidance = gr.Slider(minimum=0.01, maximum=50.0, step=0.01, value=7.5, label="Guidance")
                    inpaint_strength = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=1.0, label="Strength")
                with gr.Row():
                    inpaint_seed = gr.Number(value=-1, label="Seed", interactive=True,precision=0)
                    inpaint_button = gr.Button("Inpaint")
            with gr.Column():
                inpaint_output = gr.Image(type="numpy", interactive=False, height=600)

    seg_input_image.upload(
        fn=lambda image: (image, []),
        inputs=[seg_input_image],
        outputs=[seg_original_image, 
                 selected_point]
    )

    seg_input_image.select(
        fn=utils.select_point,
        inputs=[seg_input_image, 
                selected_point,
                point_type],
        outputs=[seg_input_image],
    )

    seg_undo_button.click(
        fn=utils.undo_point,
        inputs=[seg_original_image, 
                selected_point],
        outputs=[seg_input_image]
    )

    seg_button.click(
        fn=run_mask,
        inputs=[seg_original_image, 
                selected_point, 
                seg_model_type, 
                device, boxes, 
                mask_thresh],
        outputs=[seg_output_image, 
                 seg_output_mask]
    )

    inpaint_button.click(
        fn=run_inpaint,
        inputs=[inpaint_image, 
                inpaint_mask_image, 
                inpaint_pos_prompt, 
                inpaint_neg_promt, 
                inpaint_model_type, 
                inpaint_lora_type, 
                inpaint_num_steps,
                inpaint_num_images, 
                inpaint_guidance, 
                inpaint_strength, 
                inpaint_seed],
        outputs=[inpaint_output]
    )