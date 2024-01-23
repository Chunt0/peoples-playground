import gradio as gr
from caption import run_BLIP, load_caption, update_caption
import utils

def caption_tab(device):
    with gr.Tab(label="BLIP Caption"):
        with gr.Row():
            gr.Markdown('# BLIP Caption a folder of images')
        with gr.Row():
            img_dir_path_1 = gr.Textbox(value="", label="Image folder to caption. Enter full path to directory.", interactive=True,)
        with gr.Row():
            seed = gr.Number(value=-1, label="Seed", precision=0)
            batch_size = gr.Number(value=1, label="Batch Size", precision=0)
            top_p = gr.Number(value=0.9, label="Top p")
            max_len = gr.Number(value=75.0, label="Max Length", precision=0)
            min_len = gr.Number(value=25.0, label="Min Length", precision=0)
        with gr.Row():
            caption_button = gr.Button("Caption Images")
    with gr.Tab(label="Fix Captions"):
        img_dict = gr.State(value={})
        current_key = gr.State(value="")
        with gr.Row():    
            gr.Markdown('# Fix BLIP Captions')
        with gr.Row():
            img_dir_path_2 = gr.Textbox(value="", label="Image folder to caption. Enter full path to directory.", interactive=True,)
            load_button = gr.Button("Load Images and Captions")
        with gr.Row(equal_height=True):
            caption = gr.Textbox(value="", interactive=True)
            image = gr.Image(type="numpy", interactive=False, height=600)
        with gr.Row():
            update_button = gr.Button("Update/Next Caption")    
            
    
    
    caption_button.click(
        fn=run_BLIP,
        inputs=[img_dir_path_1, seed, batch_size, top_p, max_len, min_len],
        outputs=[]
    )
    
    load_button.click(
        fn=load_caption,
        inputs=[img_dir_path_2, img_dict],
        outputs=[img_dict, current_key, image, caption]
    )
    
    update_button.click(
        fn=update_caption,
        inputs=[caption, img_dict, current_key],
        outputs=[image, caption, img_dict, current_key]
    )