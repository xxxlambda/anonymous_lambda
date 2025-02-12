import os
import openai
import json
from programmer import Programmer
from inspector import Inspector
from function_calling.function_calling import *
from cache.cache import *
from prompt_engineering.prompts import *
import warnings
import traceback
# import pdfkit
import zipfile
from kernel import *
from lambda_utils import *
from cache.oss import upload_oss_file

warnings.filterwarnings("ignore")

class Conversation():

    def __init__(self, config) -> None:
        self.config = config
        self.model = config['conv_model']
        self.client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url_conv_model'])
        self.programmer = Programmer(api_key=config['api_key'], model=config['programmer_model'], base_url=config['base_url_programmer'])
        self.inspector = Inspector(api_key=config['api_key'], model=config['inspector_model'], base_url=config['base_url_inspector'])
        self.messages = []
        self.chat_history = []
        self.retrieval = self.config['retrieval']
        self.kernel = CodeKernel(config['max_exe_time'])
        self.function_repository = {}
        self.my_data_cache = None
        self.max_attempts = config['max_attempts'] # maximum attempts of self-correcting
        self.error_count = 0 # error count of self-correcting
        self.repair_count = 0 # repair count of self-correcting
        self.oss_dir = None
        self.local_cache_dir = None
        self.run_code(IMPORT)


    def add_functions(self, function_lib: dict) -> None:
        self.function_repository = function_lib

    def add_data(self, data_path) -> None:
        self.my_data_cache = data_cache(data_path)

    def check_folder(self, before_files, after_files):
        new_files = set(after_files) - set(before_files)
        cloud_cache_info = []
        if new_files:
            for file in new_files:
                cache_info = upload_oss_file(self.oss_dir, os.path.join(self.local_cache_dir, file))
                cloud_cache_info.append(cache_info)
        return cloud_cache_info

    def save_conv(self):
        with open(os.path.join(self.local_cache_dir,'programmer_msg.json'), 'w') as f:
            json.dump(self.programmer.messages, f, indent=4)
            f.close()
        with open(os.path.join(self.local_cache_dir,'inspector_msg.json'), 'w') as f:
            json.dump(self.inspector.messages, f, indent=4)
            f.close()
        print(f"Conversation saved in {os.path.join(self.local_cache_dir,'programmer_msg.json')}")
        print(f"Conversation saved in {os.path.join(self.local_cache_dir, 'inspector_msg.json')}")

    def add_programmer_msg(self, message: dict):
        self.programmer.messages.append(message)

    def add_programmer_repair_msg(self, bug_code:str, error_msg:str, fix_method:str, role="user"):
        message = {"role": role, "content": CODE_FIX.format(bug_code=bug_code, error_message=error_msg, fix_method=fix_method)}
        self.programmer.messages.append(message)
    def add_inspector_msg(self, bug_code:str, error_msg:str, role="user"):
        message = {"role": role, "content": CODE_INSPECT.format(bug_code=bug_code,error_message=error_msg)}
        self.inspector.messages.append(message)

    def run_code(self,code):
        try:
            res_type, res = execute(code, self.kernel)  # error in code of kernel will result in res_type:None
        except Exception as e:
            print(f'Error when executing code in kernel: {e}')
            res_type, res = 'error', str(e)
            self.add_inspector_msg(code, str(e))
        return res_type, res

    def rendering_code(self):
        for i in range(len(self.programmer.messages)-1, 0, -1):
            if self.programmer.messages[i]["role"] == "assistant":
                is_python, code = extract_code(self.programmer.messages[i]["content"])
                if is_python:
                    return code
        return None

    # def human_loop(self, code):
    #     self.programmer.messages.append({"role": "user", "content": HUMAN_LOOP.format(code=code)})
    #
    #     result = self.run_code(code)
    #     exec_result = f"Execution result of user's code:\n{result}."
    #     self.programmer.messages.append({"role": "system", "content": exec_result})
    #     return response

    def show_data(self) -> pd.DataFrame:
        return self.my_data_cache.data

    def document_generation(self):  # use conv model to generate report.
        print("Report generating...")
        self.messages = self.programmer.messages + [{"role": "user", "content": Academic_Report}]
        # self.programmer.messages.append({"role": "user", "content": DOCUMENT})
        report = self.call_chat_model(max_tokens=self.config["max_token_conv_model"]).choices[0].message.content
        self.messages.append({"role": "assistant", "content": report})
        #self.local_cache_dir = 'xxx'
        mkd_path = os.path.join(self.local_cache_dir,'report.md')
        # pdf_path = os.path.join(self.local_cache_dir,'report.pdf')
        # zip_path = os.path.join(self.local_cache_dir,'report.zip')
        with open(mkd_path, "w") as f:
            f.write(report)
            f.close()
        # pdfkit.from_file(mkd_path, pdf_path)
        # with zipfile.ZipFile(zip_path, 'w') as zipf:
        #     zipf.write('report.md')
        #     zipf.write('report.pdf')
        return mkd_path

    def export_code(self):
        print("Exporting notebook...")
        notebook_path = os.path.join(self.local_cache_dir, 'notebook.ipynb')
        try:
            self.kernel.export(notebook_path)
        except Exception as e:
            print(f"An error occurred when exporting notebook: {e}")
        return notebook_path

    def call_chat_model(self, max_tokens, functions=None, include_functions=False, ) -> dict:
        params = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": None if max_tokens is None else max_tokens
        }

        if include_functions:
            params["functions"] = functions
            #params["function_call"] = "auto"

        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            self.messages = [self.messages[0], self.messages[-1]]
            params = {
                "model": self.model,
                "messages": self.messages
            }
            print(f"Occurs error of calling chat model: {e}")
            return self.client.chat.completions.create(**params)

    def clear(self):
        self.messages = []
        self.programmer.clear()
        self.inspector.clear()
        self.kernel.shutdown()
        del self.kernel
        self.kernel = CodeKernel()
        self.my_data_cache = None
        self.my_data_description = None
        self.oss_dir = None
        self.local_cache_dir = None

    def stream_workflow(self, chat_history, function_lib: dict=None, code=None) -> object:
        try:
            chat_history[-1][1] = ""
            if not self.my_data_cache and code is None:  #todo: remove this restriction
                for message in self.programmer._call_chat_model_streaming():
                    chat_history[-1][1] += message
                    yield chat_history
                #chat_history += "\nNote, no data found. Please upload the data manually to start your task."
                yield chat_history
                final_response = chat_history[-1][1] #''.join(chat_history[-1][1])
                self.add_programmer_msg({"role": "assistant", "content": final_response})

            else:
                if not code:
                    #_, _ = execute(IMPORT, self.kernel) #todo install packages

                    prog_response1_content = ''
                    for message in self.programmer._call_chat_model_streaming(retrieval=self.retrieval, kernel=self.kernel):
                        chat_history[-1][1] += message
                        prog_response1_content += message
                        yield chat_history

                    self.add_programmer_msg({"role": "assistant", "content": prog_response1_content})
                else:
                    prog_response1_content = HUMAN_LOOP.format(code=code)
                    self.add_programmer_msg({"role": "user", "content": prog_response1_content})
                is_python, code = extract_code(prog_response1_content)
                print("is_python:",is_python)

                if is_python:
                    chat_history[-1][1] += '\n🖥️ Execute code...\n'
                    yield chat_history
                    before_files = os.listdir(self.local_cache_dir)
                    res_type, res = self.run_code(code)
                    if res_type and res_type != 'error' or not res:  # no error in code   not res = res is None
                        after_files = os.listdir(self.local_cache_dir)
                        cloud_cache_info = self.check_folder(before_files, after_files)
                        link_info = '\n'.join([f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                        ('.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})" for info in
                                   cloud_cache_info])
                        print("res:", res)

                        chat_history[-1][1] += f"\n✅ Execution result:\n{res}\n\n"
                        yield chat_history
                        self.add_programmer_msg({"role": "user", "content": RESULT_PROMPT.format(res)})

                        prog_response2 = ''
                        for message in self.programmer._call_chat_model_streaming():
                            chat_history[-1][1] += message
                            prog_response2 += message
                            yield chat_history

                        self.add_programmer_msg({"role": "assistant", "content": prog_response2})

                        chat_history[-1][1] += f"\n{link_info}"
                        yield chat_history

                    #elif not res_type and res:  #todo: problems related the image output in the result.

                    # self-correcting
                    else:
                        self.error_count += 1
                        round = 0
                        while res and round < self.max_attempts: #max 5 round
                            chat_history[-1][1] = f'⭕ Execution error, try to repair the code, attempts: {round+1}....\n'
                            yield chat_history
                            self.add_inspector_msg(code, res)
                            if round == 3:
                                insp_response1_content = "Try other packages or methods."
                            else:
                                insp_response1 = self.inspector._call_chat_model()
                                insp_response1_content = insp_response1.choices[0].message.content
                            self.inspector.messages.append({"role": "assistant", "content": insp_response1_content})

                            self.add_programmer_repair_msg(code, res, insp_response1_content)

                            prog_response1_content = ''
                            for message in self.programmer._call_chat_model_streaming():
                                chat_history[-1][1] += message
                                prog_response1_content += message
                                yield chat_history
                            chat_history[-1][1] += '\n🖥️ Execute code...\n'
                            yield chat_history

                            self.add_programmer_msg({"role": "assistant", "content": prog_response1_content})
                            is_python, code = extract_code(prog_response1_content)
                            if is_python:
                                before_files = os.listdir(self.local_cache_dir)
                                res_type, res = self.run_code(code)
                                if res_type and res_type != 'error' or not res: # execute successfully
                                    self.repair_count += 1
                                    break
                            round += 1
                        if round == self.max_attempts: # maximum retry
                            return prog_response1_content + "\nSorry, I can't fix the code, can you help me to modified it or give some suggestions?"
                            #self.add_programmer_msg({"role": "user", "content": "You can not tackle the error, you should ask me to give suggestions or modify the code."})

                        # revision successful
                        after_files = os.listdir(self.local_cache_dir)
                        cloud_cache_info = self.check_folder(before_files, after_files)
                        link_info = '\n'.join([f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                            ('.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})" for info in
                                               cloud_cache_info])
                        print("res:", res)
                        chat_history[-1][1] += f"\n✅ Execution result:\n{res}\n\n"
                        yield chat_history

                        self.add_programmer_msg({"role": "user", "content": RESULT_PROMPT.format(res)})

                        prog_response2 = ''
                        for message in self.programmer._call_chat_model_streaming():
                            chat_history[-1][1] += message
                            prog_response2 += message
                            yield chat_history
                        self.add_programmer_msg({"role": "assistant", "content": prog_response2})
                        chat_history[-1][1] += f"\n{link_info}"
                        yield chat_history

                    # else: #save file, figure..
                    #     after_files = os.listdir(self.local_cache_dir)
                    #     cloud_cache_info = self.check_folder(before_files, after_files)
                    #     link_info = '\n'.join(
                    #         [f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                    #             (
                    #             '.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})"
                    #          for info in
                    #          cloud_cache_info])
                    #     print("res:", res)
                    #     self.add_programmer_msg({"role": "system", "content": FIG_PROMPT})
                    #     prog_response2 = self.programmer._call_chat_model().choices[0].message.content
                    #     final_response = prog_response1_content + "\n" + prog_response2
                    #     final_response += f"\n{link_info}"

                else:
                    chat_history[-1][1] += "\nNo code detected or code is not python code." #todo : delete printing this
                    yield chat_history
                    final_response = prog_response1_content + "\nNo code detected or code is not python code."
                    if self.programmer.messages[-1]["role"] == "assistant":
                        self.programmer.messages[-1]["content"] = final_response

        except Exception as e:
            chat_history[-1][1] += "\nSorry, there is an error in the program, please try again."
            print(f"An error occurred: {e}")
            traceback.print_exc()
            if self.programmer.messages[-1]["role"] == "user":
                self.programmer.messages.append({"role": "assistant", "content": f"An error occurred in program: {e}"})
            #return "Sorry, there is an error in the program, please try again."

        #return final_response


    # a previous code without streaming
    def run_workflow(self,function_lib: dict=None, code=None) -> object:
        try:
            if not self.my_data_cache and code is None:
                final_response = self.programmer._call_chat_model().choices[0].message.content
                self.add_programmer_msg({"role": "assistant", "content": final_response})
            else:
                if not code:
                    #_, _ = execute(IMPORT, self.kernel) #todo prompt of install packages
                    prog_response1 = self.programmer._call_chat_model()
                    prog_response1_content = prog_response1.choices[0].message.content
                    self.add_programmer_msg({"role": "assistant", "content": prog_response1_content})
                else:
                    prog_response1_content = HUMAN_LOOP.format(code=code)
                    self.add_programmer_msg({"role": "user", "content": prog_response1_content})
                    self.add_programmer_msg({"role": "assistant", "content": "I got the code, let me execute it."})
                is_python, code = extract_code(prog_response1_content)
                print("is_python:",is_python)

                if is_python:
                    before_files = os.listdir(self.local_cache_dir)
                    res_type, res = self.run_code(code)
                    if res_type and res_type != 'error' or not res: # no error in code   not res = res is None
                        after_files = os.listdir(self.local_cache_dir)
                        cloud_cache_info = self.check_folder(before_files, after_files)
                        link_info = '\n'.join([f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                        ('.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})" for info in
                                   cloud_cache_info])
                        print("res:", res)

                        self.add_programmer_msg({"role": "user", "content": RESULT_PROMPT.format(res)})

                        prog_response2 = self.programmer._call_chat_model().choices[0].message.content

                        final_response = prog_response1_content + "\n" + f"Execution result:\n{res}\n\n{prog_response2}"
                        final_response += f"\n{link_info}"

                    #elif not res_type and res: #error occurs in code, only image
                    else:
                        self.error_count += 1
                        round = 0
                        while res and round < self.max_attempts: #max 5 round
                            self.add_inspector_msg(code, res)
                            if round == 3:
                                insp_response1_content = "Try other packages or methods."
                            else:
                                insp_response1 = self.inspector._call_chat_model()
                                insp_response1_content = insp_response1.choices[0].message.content
                            self.inspector.messages.append({"role": "assistant", "content": insp_response1_content})
                            #self.add_programmer_msg(modify_msg)

                            self.add_programmer_repair_msg(code, res, insp_response1_content)
                            prog_response1 = self.programmer._call_chat_model()
                            prog_response1_content = prog_response1.choices[0].message.content
                            self.add_programmer_msg({"role": "assistant", "content": prog_response1_content})
                            is_python, code = extract_code(prog_response1_content)
                            if is_python:
                                before_files = os.listdir(self.local_cache_dir)
                                res_type, res = self.run_code(code)
                                if res_type or not res: # execute successfully
                                    self.repair_count += 1
                                    break
                            round += 1
                        if round == self.max_attempts: # max retry
                            # self.add_programmer_msg({"role": "assistant", "content": "Sorry, I can't fix the code, can you help me to modified it or give some suggestions?"})
                            return prog_response1_content + "\nSorry, I can't fix the code, can you help me to modified it or give some suggestions?"

                        # revision successful
                        after_files = os.listdir(self.local_cache_dir)
                        cloud_cache_info = self.check_folder(before_files, after_files)
                        link_info = '\n'.join([f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                            ('.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})" for info in
                                               cloud_cache_info])
                        print("res:", res)
                        if 'head()' in code:
                            self.messages.append({"role": "user", "content": MKD_PROMPT.format(res)})
                            mkd_res = self.call_chat_model(max_tokens=512).choices[0].message.content
                            res = mkd_res
                            self.messages.pop()

                        self.add_programmer_msg({"role": "user", "content": RESULT_PROMPT.format(res)})

                        prog_response2 = self.programmer._call_chat_model().choices[0].message.content
                        # todo: delete code in second response
                        # prog_response2 = remove_code_blocks(prog_response2)
                        self.add_programmer_msg({"role": "assistant", "content": prog_response2})
                        # final_response = prog_response1_content + "\n" + prog_response2
                        final_response = prog_response1_content + "\n" + f"Execution result:\n{res}\n\n{prog_response2}"
                        final_response += f"\n{link_info}"

                    # else: #save file, figure..
                    #     after_files = os.listdir(self.local_cache_dir)
                    #     cloud_cache_info = self.check_folder(before_files, after_files)
                    #     link_info = '\n'.join(
                    #         [f"![{info['file_name']}]({info['download_link']})" if info['file_name'].endswith(
                    #             (
                    #             '.jpg', '.jpeg', '.png', '.gif')) else f"[{info['file_name']}]({info['download_link']})"
                    #          for info in
                    #          cloud_cache_info])
                    #     print("res:", res)
                    #     self.add_programmer_msg({"role": "system", "content": FIG_PROMPT})
                    #     prog_response2 = self.programmer._call_chat_model().choices[0].message.content
                    #     final_response = prog_response1_content + "\n" + prog_response2
                    #     final_response += f"\n{link_info}"


                else:
                    final_response = prog_response1_content + "\n" + "No code detected or code is not python code."
                    if self.programmer.messages[-1]["role"] == "assistant":
                        self.programmer.messages[-1]["content"] = final_response

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            if self.programmer.messages[-1]["role"] == "user":
                self.programmer.messages.append({"role": "assistant", "content": f"An error occurred in program: {e}"})
            return "Sorry, there is an error in the program, please try again."

        return final_response

# function_lib = {
#
# }





