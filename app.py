import gradio as gr
import torch
from txt2img_gui import txt2img_tab
from segment_gui import segment_tab
from train_gui import train_tab
from caption_gui import caption_tab


with gr.Blocks(title="P.E.O.P.L.E's PLAYGROUND", theme=gr.themes.Monochrome()) as demo: 
    with gr.Row():
        gr.Markdown("# P.E.O.P.L.E's PLAYGROUND")
        device = gr.State(value=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    with gr.Tab("Txt2Img"):
        txt2img_tab()                
    with gr.Tab("Segment/Inpaint"):
        segment_tab(device)
    with gr.Tab("Caption"):
        caption_tab(device)
    with gr.Tab("Train"):
        train_tab()
  
demo.launch()