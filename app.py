import os.path
import shutil
import sys
import traceback
import gradio as gr
import json
import random
import time
import uuid
import pandas as pd
from conversation import Conversation
from cache.oss import init_dir, init_oss, upload_oss_file, get_download_link
from cache.cache import data_cache
from prompt_engineering.prompts import *
from kernel import CodeKernel
import yaml
# from config import conv_model, streaming
import atexit
import signal

class app():
    def __init__(self, config_path='config.yaml'):
        print("Load config: ", config_path)
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.conv = Conversation(self.config)

        self.conv.programmer.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    def open_board(self):
        return self.conv.show_data()

    def cache_file(self,file):
        try:
            self.oss_file_dir, self.local_cache_dir = init_dir(self.config['cache_dir'])
            init_oss(self.config['oss_endpoint'], self.config['oss_access_key_id'], self.config['oss_access_secret'], self.config['oss_bucket_name'], self.config['expired_time'])
            self.conv.oss_dir, self.conv.local_cache_dir = self.oss_file_dir, self.local_cache_dir
            file_info = upload_oss_file(self.oss_file_dir, file)
            local_file_path = os.path.join(self.local_cache_dir, file_info['file_name'])
            shutil.copy2(file, local_file_path)
            self.conv.add_data(file)
            gen_info = self.conv.my_data_cache.get_description()
            self.conv.programmer.messages[0]["content"] = (PROGRAMMER_PROMPT.format(working_path=self.local_cache_dir) +
                                                f"\nNow, user uploads the datasets in {local_file_path}\n, and here is the general information of the dataset:\n {gen_info}")
            if self.conv.retrieval:
                self.conv.programmer.messages[0]["content"] += KNOWLEDGE_INTEGRATION_SYSTEM
            print(self.conv.programmer.messages[0]["content"])

            # self.conv.run(function_lib)
            # self.conv.messages.append({"role": "assistant", "content": response})
            # chat_history.append((message, response))
        except Exception as e:
            traceback.print_exc()
            print(f"Uploaded file error: {e}")

    # def show_data(self):
    #     return self.conv.show_data()

    def rendering_code(self):
        return self.conv.rendering_code()

    # def report_test(self):
    #     with open("/Users/stephensun/Desktop/pypro/dsagent_ci/cache/conv_cache/2c23cf4b-31a8-4aeb-8689-13b9fbdfb7fc-2024-04-29/programmer_msg.json", "r") as f:
    #         msg = json.load(f)
    #         print(msg)
    #     self.conv.programmer.messages = msg
    #     down_path = self.conv.document_generation()
    #     return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Report", value=down_path, visible=True)]

    def generate_report(self):
        down_path = self.conv.document_generation()
        return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Report", value=down_path, visible=True)]

    def export_code(self):
        down_path = self.conv.export_code()
        return [gr.Button(visible=False), gr.DownloadButton(label=f"Download Notebook", value=down_path, visible=True)]

    def down_report(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]
    def down_notebook(self):
        return [gr.Button(visible=True), gr.DownloadButton(visible=False)]

    # def human_loop(self, code):
    #     result = self.conv.run_workflow(code)
    #     self.conv.programmer.messages.append({"role": "assistant", "content": response})
    #     return response
    def call_llm(self, message, chat_history, code=None):
        '''
        :param message: input of user
        :param chat_history: the history of whole dialogue. But this variable was only used to show in the UI (not record in LLM)
        :return: chat_history in the Chatbot component
        '''
        if not code:
            self.conv.programmer.messages.append({"role": "user", "content": message})
        else:
            message = code
        response = self.conv.run_workflow(function_lib=None,code=code)
        chat_history.append((message, response))
        self.conv.chat_history = chat_history
        # chat_history = self.conv.messages
        # print(chat_history)
        return "", chat_history  # return "" to let msg box be clear

    # def load_chat_history(self):
    #     # filtered_messages = [msg for msg in self.conv.programmer.messages if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]]
    #     # # [{"role": "user", "content": "qqq"},{"role": "assistant", "content": "aaa"}]
    #     # chat_history = [(filtered_messages[i].get('content'), filtered_messages[i + 1].get('content')) for i in range(0, len(filtered_messages), 2)]
    #     return self.conv.chat_history

    def save_dialogue(self, chat_history):
        self.conv.save_conv()
        with open(os.path.join(self.local_cache_dir,'system_dialogue.json'), 'w') as f:
            json.dump(chat_history, f, indent=4)
            f.close()
        print(f"Dialogue saved in {os.path.join(self.local_cache_dir,'system_dialogue.json')}.")

    def load_dialogue(self, json_file):
        with open(json_file, 'r') as f:
            chat_history = json.load(f)
            f.close()
        self.conv.chat_history = chat_history
        return chat_history

    def clear_all(self, message, chat_history):
        self.conv.clear()
        return "", []

my_app = app()

with gr.Blocks(theme=gr.themes.Soft(), css='front_end/css.css', js='front_end/javascript.js') as demo:
    # get history for chatbot.
    #history = my_app.load_dialogue("system_dialogue.json") # load history in previous conversation by indicate system_dialogue.json
    chatbot = gr.Chatbot(value=my_app.conv.chat_history, height=600, label="LAMBDA", avatar_images=["front_end/user.jpg", "front_end/lambda.jpg"], show_copy_button=True)
    with gr.Group():
        with gr.Row():
            upload_btn = gr.UploadButton(label="Upload Data", file_types=["csv", "xlsx"], scale=1)
            # datasets = gr.File(file_types=["csv", "xlsx"], elem_id="file_upload",height=80)
            msg = gr.Textbox(show_label=False, placeholder="Sent message to LLM", scale=6)
            submit = gr.Button("Submit", scale=1)
    with gr.Row():
        board = gr.Button(value="Show/Update DataFrame", elem_id="df_btn", elem_classes="df_btn")
        export_notebook = gr.Button(value="Notebook")
        down_notebook = gr.DownloadButton("Download Notebook", visible=False)
        generate_report = gr.Button(value="Generate Report")
        down_report = gr.DownloadButton("Download Report",visible=False)

        edit = gr.Button(value="Edit Code",elem_id="ed_btn", elem_classes="ed_btn")
        save = gr.Button(value="Save Dialogue")
        clear = gr.ClearButton(value="Clear All")

    with gr.Group():
        with gr.Row(visible=False, elem_id="ed",elem_classes="ed"):
            code = gr.Code(label="Code",scale=6)
            code_btn = gr.Button("Submit Code",scale=1)
    code_btn.click(fn=my_app.call_llm, inputs=[msg, chatbot, code], outputs=[msg, chatbot])


    df = gr.Dataframe(visible=False,elem_id="df",elem_classes="df")

    upload_btn.upload(fn=my_app.cache_file, inputs=upload_btn)
    if my_app.config['streaming']:
        def chat_streaming(message, chat_history, code=None):
            if not code:
                my_app.conv.programmer.messages.append({"role": "user", "content": message})
            else:
                message = code
            #my_app.conv.stream_workflow(function_lib=None, code=code)
            return "", chat_history + [[message, None]]

        msg.submit(chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
            my_app.conv.stream_workflow, chatbot, chatbot
        )
        submit.click(chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
            my_app.conv.stream_workflow, chatbot, chatbot
        )
    else:
        msg.submit(my_app.call_llm, [msg, chatbot], [msg, chatbot])
        submit.click(my_app.call_llm, [msg, chatbot], [msg, chatbot])
    board.click(my_app.open_board, inputs=[], outputs=df)
    edit.click(my_app.rendering_code, inputs=None, outputs=code)
    export_notebook.click(my_app.export_code, inputs=None, outputs=[export_notebook, down_notebook])
    down_notebook.click(my_app.down_notebook, inputs=None, outputs=[export_notebook, down_notebook])
    generate_report.click(my_app.generate_report,inputs=None,outputs=[generate_report,down_report])
    down_report.click(my_app.down_report,inputs=None,outputs=[generate_report,down_report])
    save.click(my_app.save_dialogue, inputs=chatbot)
    clear.click(fn=my_app.clear_all, inputs=[msg, chatbot], outputs=[msg, chatbot])
    #demo.load(fn=my_app.load_chat_history, inputs=None, outputs=chatbot)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0",server_port=8001, debug=True, share=True)
