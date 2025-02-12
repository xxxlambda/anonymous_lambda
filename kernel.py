import queue
# import streamlit as st
import base64
from io import BytesIO
import re
import os
import jupyter_client
from PIL import Image
from subprocess import PIPE
from pprint import pprint
import nbformat as nbf

IPYKERNEL = os.environ.get('IPYKERNEL', 'dsagent')

# client = get_client()


class CodeKernel(object):
    def __init__(self,
                 kernel_name='kernel',
                 kernel_id=None,
                 kernel_config_path="",
                 python_path=None,
                 ipython_path=None,
                 init_file_path="./startup.py",
                 verbose=1,
                 max_exe_time=6000):

        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.kernel_config_path = kernel_config_path
        self.python_path = python_path
        self.ipython_path = ipython_path
        self.init_file_path = init_file_path
        self.executed_cells = []
        self.verbose = verbose
        self.interrupt_signal = False
        self.max_exe_time = max_exe_time


        if python_path is None and ipython_path is None:
            env = None
        else:
            env = {"PATH": self.python_path + ":$PATH", "PYTHONPATH": self.python_path}

        # Initialize the backend kernel
        self.kernel_manager = jupyter_client.KernelManager(kernel_name=IPYKERNEL,
                                                           connection_file=self.kernel_config_path,
                                                           exec_files=[self.init_file_path],
                                                           env=env)
        if self.kernel_config_path:
            self.kernel_manager.load_connection_file()
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_config_path))
        else:
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_manager.connection_file))

        if verbose:
            pprint(self.kernel_manager.get_connection_info())

        # Initialize the code kernel
        self.kernel = self.kernel_manager.blocking_client()
        # self.kernel.load_connection_file()
        self.kernel.start_channels()
        print("Code kernel started.")

    def execute(self, code):
        self.kernel.execute(code)
        try:
            shell_msg = self.kernel.get_shell_msg(timeout=self.max_exe_time)
            io_msg_content = self.kernel.get_iopub_msg(timeout=self.max_exe_time)['content']
            msg_out_list = []
            while True:
                msg_out = io_msg_content
                if 'name' in msg_out and msg_out['name'] != 'stderr':
                    msg_out_list.append(msg_out)
                elif 'data' in msg_out and 'image/png' in msg_out['data']:
                    msg_out_list.append(msg_out)
                ### Poll the message
                try:
                    io_msg_content = self.kernel.get_iopub_msg(timeout=self.max_exe_time)['content']
                    if 'execution_state' in io_msg_content and io_msg_content['execution_state'] == 'idle':
                        msg_out_list.append(msg_out)
                        break
                except queue.Empty:
                    break
            # msg_out = '\n'.join(msg_out_list)
            return shell_msg, msg_out_list
        except Exception as e:
            print(e)
            return None
        
    def execute_code_(self, code):
        msg_id = self.kernel.execute(code)

        # Get the output of the code
        msg_list = []
        while True:
            try:
                iopub_msg = self.kernel.get_iopub_msg(timeout=self.max_exe_time)
                msg_list.append(iopub_msg)
                if iopub_msg['msg_type'] == 'status' and iopub_msg['content'].get('execution_state') == 'idle':
                    break
            except:
                if self.interrupt_signal:
                    self.kernel_manager.interrupt_kernel()
                    self.interrupt_signal = False
                continue

        all_output = []
        for iopub_msg in msg_list:
            if iopub_msg['msg_type'] == 'stream':
                if iopub_msg['content'].get('name') == 'stdout':
                    output = iopub_msg['content']['text']
                    all_output.append(('stdout', output))
            elif iopub_msg['msg_type'] == 'execute_result':
                if 'data' in iopub_msg['content']:
                    if 'text/plain' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/plain']
                        all_output.append(('execute_result_text', output))
                    if 'text/html' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/html']
                        all_output.append(('execute_result_html', output))
                    if 'image/png' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/png']
                        all_output.append(('execute_result_png', output))
                    if 'image/jpeg' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/jpeg']
                        all_output.append(('execute_result_jpeg', output))
            elif iopub_msg['msg_type'] == 'display_data':
                if 'data' in iopub_msg['content']:
                    if 'text/plain' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/plain']
                        all_output.append(('display_text', output))
                    if 'text/html' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['text/html']
                        all_output.append(('display_html', output))
                    if 'image/png' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/png']
                        all_output.append(('display_png', output))
                    if 'image/jpeg' in iopub_msg['content']['data']:
                        output = iopub_msg['content']['data']['image/jpeg']
                        all_output.append(('display_jpeg', output))
            elif iopub_msg['msg_type'] == 'error':
                if 'traceback' in iopub_msg['content']:
                    output = '\n'.join(iopub_msg['content']['traceback'])
                    all_output.append(('error', output))

        return all_output


    def export(self, file_path):
        nb = nbf.v4.new_notebook()
        nb.cells = self.executed_cells
        print("cell: ",nb.cells)
        with open(file_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Notebook exported to {file_path}")

    def execute_interactive(self, code, verbose=False):
        shell_msg = self.kernel.execute_interactive(code)
        if shell_msg is queue.Empty:
            if verbose:
                print("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)

        return shell_msg

    def inspect(self, code, verbose=False):
        msg_id = self.kernel.inspect(code)
        shell_msg = self.kernel.get_shell_msg(timeout=30)
        if shell_msg is queue.Empty:
            if verbose:
                print("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)

        return shell_msg

    def get_error_msg(self, msg, verbose=False) -> str | None:
        if msg['content']['status'] == 'error':
            try:
                error_msg = msg['content']['traceback']
            except:
                try:
                    error_msg = msg['content']['traceback'][-1].strip()
                except:
                    error_msg = "Traceback Error"
            if verbose:
                print("Error: ", error_msg)
            return error_msg
        return None

    def check_msg(self, msg, verbose=False):
        status = msg['content']['status']
        if status == 'ok':
            if verbose:
                print("Execution succeeded.")
        elif status == 'error':
            for line in msg['content']['traceback']:
                if verbose:
                    print(line)

    def shutdown(self):
        # Shutdown the backend kernel
        self.kernel_manager.shutdown_kernel(now=True)
        print("Backend kernel shutdown.")
        # Shutdown the code kernel
        self.kernel.shutdown()
        print("Code kernel shutdown.")

    def restart(self):
        # Restart the backend kernel
        self.kernel_manager.restart_kernel()
        print("Backend kernel restarted.")

    def start(self):
        # Initialize the code kernel
        self.kernel = self.kernel_manager.blocking_client()
        # self.kernel.load_connection_file()
        self.kernel.start_channels()
        print("Code kernel started.")

    def interrupt(self):
        # Interrupt the backend kernel
        self.kernel_manager.interrupt_kernel()
        print("Backend kernel interrupted.")

    def is_alive(self):
        return self.kernel.is_alive()


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def clean_ansi_codes(input_string):
    ansi_escape = re.compile(r'(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', input_string)


def execute(code, kernel: CodeKernel) -> tuple[str, str | Image.Image]:
    res = ""
    res_type = None
    # code = code.replace("<|observation|>", "")
    # code = code.replace("<|assistant|>interpreter", "")
    # code = code.replace("<|assistant|>", "")
    # code = code.replace("<|user|>", "")
    # code = code.replace("<|system|>", "")
    msg, output_list = kernel.execute(code)

    if msg['metadata']['status'] == "timeout":
        return res_type, 'Timed out'
    elif msg['metadata']['status'] == 'error':
        return res_type, clean_ansi_codes('\n'.join(kernel.get_error_msg(msg, verbose=True))) #todo: change the res_type to error

    if len(output_list) > 1:
        res_list = []
        for output in output_list:
            if 'text' in output:
                res_type = "text"
                res = output['text']
            elif 'data' in output:
                for key in output['data']:
                    if 'text/plain' in key:
                        res_type = "text"
                        res = output['data'][key]
                    elif 'image/png' in key:
                        res_type = "image"
                        res = output['data'][key]
                        break

            if res_type == "image":
                #return res_type, b64_2_img(res)
                res_list.append(b64_2_img(res))
            elif res_type == "text" or res_type == "traceback": #traceback is error?
                #res = res
                res_list.append(res)
        if len(res_list) > 1 and res_list[-1] == res_list[-2]:
            res_list.pop()

        res = '\n'.join(res_list)
        return res_type, res
    else:
        output = output_list[0]
        if 'text' in output: #output is a dict, what situation is 'text' in output?
            res_type = "text"
            res = output['text']
        elif 'data' in output:
            for key in output['data']:
                if 'text/plain' in key: #key = 'text/plain'
                    res_type = "text"
                    res = output['data'][key]
                elif 'image/png' in key: # image will be present as html file?
                    res_type = "image"
                    res = output['data'][key]
                    break

        if res_type == "image":
            return res_type, b64_2_img(res)
        elif res_type == "text" or res_type == "traceback":  # traceback is error?
            res = res
        return res_type, res


# @st.cache_resource
def get_kernel():
    kernel = CodeKernel()
    return kernel


if __name__ == '__main__':
    kernel = CodeKernel()
    table_code = """
    import pandas as pd
    data = pd.read_csv('/Users/stephensun/Desktop/pypro/LAMBDA/cache/conv_cache/c1829f26-f6b3-4cbc-96a7-88aef2c33df2-2024-08-13/wine.csv')
    data.head()
    """
    img_code = """
    import matplotlib.pyplot as plt
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 6]
    
    plt.plot(x, y)
    
    plt.title('Simple Line Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    """
    res_type, res = execute(img_code, kernel)
    print(res_type,res)
    # kernel.export("my_notebook.ipynb")

