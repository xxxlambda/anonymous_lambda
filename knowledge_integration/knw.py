import inspect
import textwrap


KNW_INJECTION = {}

class knw:
    def __init__(self):
        self.name = 'knowledge_integration'
        self.description = 'Integrate knowledge into the LLM.'
        self.core_function = 'core_function'
        self.test_case = 'test_case'
        self.runnable_function = 'runnable_function'
        self.mode = 'full'
        self.method_code = {}

    def get_core_function(self):
        """
        Core function of the knowledge integration.
        """
        function_name = self.core_function
        core_function = getattr(self, function_name, None)
        return textwrap.dedent(core_function())

        # return self.method_code[self.core_function]

    def get_runnable_function(self):
        """
        Runnable function of the knowledge integration.
        """
        function_name = self.runnable_function
        runnable_function = getattr(self, function_name, None)
        return textwrap.dedent(runnable_function())
        #return self.method_code[self.runnable_function]

    def get_all_code(self):
        return self.get_core_function(), self.get_runnable_function()
        #return "Core code:" + self.get_core_function() + "\nOther function code" + self.get_runnable_function()

    def get_test_case(self):
        """
        Test case for the knowledge integration.
        """
        return self.method_code[self.test_case]

    def get_internal_function(self):
        """
        All other functions of the core function.
        """
        internal_code = ""
        for name, code in self.method_code.items():
            if name not in [self.core_function, self.test_case]:
                internal_code += f"{code}\n"
        return internal_code


    def get_function_code(self, function_name):
        function = getattr(self, function_name, None)
        if function is None:
            logger.warning("Method not found.")
        else:
            inspect_function = inspect.getsource(function)
            return inspect_function.replace('self,', '').replace('self.', '').replace('self','')

    def get_all_function_code(self):
        all_code = "```python"
        for name, code in self.method_code.items():
            all_code += f"\n{code}\n"
        return all_code+"```"

    def get_all_function(self):
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        self.method_code = {name: self.get_function_code(name) for name, _ in methods if name not in ['__init__', 'get_all_function', 'get_all_function_code', 'get_core_function', 'get_function_code', 'get_test_case', 'get_internal_function', 'get_fixed_function']}
        return self.method_code

    def get_fix_function(self):
        return self.method_code

    def remove_outer_indentation(code_str):  # delete the outer indentation
        lines = code_str.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return code_str
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        aligned_lines = [line[min_indent:] for line in lines]

        return '\n'.join(aligned_lines)


if __name__ == '__main__':
    kwn = knw()
    # print(kwn.get_entrance_function()) #todo : problem in indent
    print(kwn.get_all_function_code())
    # print(instantiate_subclasses(knw))