import gradio as gr
import json
from datetime import datetime

with open("log.json", 'r') as file:
    data = json.load(file) 

tasks = ["consult", "dataprep", "training", "eval", "admin", "other"]

def calculate_time_difference(start_str, end_str):
    date_format = '%Y:%m:%d:%H:%M:%S'
    start_time = datetime.strptime(start_str, date_format)
    end_time = datetime.strptime(end_str, date_format)
    elapsed_time =  start_time - end_time
    elapsed_hours = elapsed_time.total_seconds() / 3600
    return elapsed_hours

def load_proj(new_proj_name, new_proj_checkbox, project):
    if new_proj_checkbox:
        now = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        data[new_proj_name]= {"time":"0", "log":f"created|{now}","note":""}
        with open("log.json", 'w') as file:
            json.dump(data, file, indent=4)
        return new_proj_name, data[new_proj_name]["log"]
    return project, data[project]["log"]

def start_time(loaded_proj_name, task):
    log = data[loaded_proj_name]["log"]
    split_log = log.split("\n")
    if "start" in split_log[-1]:
        print(f"Last log was a start time: {split_log[-1]}")
        return log
    else:
        now = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        log = log + f"\nstart|{task}|{now}"
        data[loaded_proj_name]["log"] = log
        with open("log.json", 'w') as file:
            json.dump(data, file, indent=4)
        return log

def stop_time(loaded_proj_name, task):
    log = data[loaded_proj_name]["log"]
    split_log = log.split("\n")
    if "stop" in split_log[-1]:
        print(f"Last log was a stop time: {split_log[-1]}")
        return log
    else:
        now = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        before = split_log[-1].split("|")[-1]
        elapsed_time = calculate_time_difference(now, before)
        data[loaded_proj_name]["time"] = float(data[loaded_proj_name]["time"])+elapsed_time
        log = log + f"\nstop|{task}|{now}"
        data[loaded_proj_name]["log"] = log
        with open("log.json", 'w') as file:
            json.dump(data, file, indent=4)
        return log

def record_note(loaded_proj_name, task, note):
    now = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    note_recorded = f"Note recorded {now}"
    note = data[loaded_proj_name]["note"]+f"{task}|{note}|{now}\n"
    data[loaded_proj_name]["note"] = note
    with open("log.json", 'w') as file:
        json.dump(data, file, indent=4)
    return note_recorded

with gr.Blocks(title="time-tracker", theme=gr.themes.Monochrome()) as demo:
    with gr.Row():
        gr.Markdown("# Project Tracker")
        with gr.Column():
            new_proj_checkbox = gr.Checkbox(value=False, label="New Project?")
            new_proj_name = gr.Textbox(value="", label="New Project Name")
        with gr.Column():   
            project = gr.Dropdown(choices=data, interactive=True, label="Select Project")
            load_button = gr.Button("Load Project")        
    with gr.Row():
        with gr.Column():
            loaded_proj_name = gr.Textbox(value="", interactive=False, label="Current Project")
            task = gr.Dropdown(choices=tasks, interactive=True, label="Select Task")
        with gr.Column():
            loaded_proj_log = gr.Textbox(value="", interactive=False, label="Log", max_lines=7)    
    with gr.Row():
        start_button = gr.Button("Start Timer")
        stop_button = gr.Button("Stop Timer")
    with gr.Row():
        note = gr.Textbox(value="", label="Note")
        note_recorded = gr.Text(value="")
        note_button = gr.Button("Record Note")

    load_button.click(
        fn=load_proj,
        inputs=[new_proj_name, new_proj_checkbox,project],
        outputs=[loaded_proj_name, loaded_proj_log]
    )

    start_button.click(
        fn=start_time,
        inputs=[loaded_proj_name, task],
        outputs=[loaded_proj_log]
    )

    stop_button.click(
        fn=stop_time,
        inputs=[loaded_proj_name, task],
        outputs=[loaded_proj_log]
    )

    note_button.click(
        fn=record_note,
        inputs=[loaded_proj_name, task, note],
        outputs=[note_recorded]
    )

demo.launch()