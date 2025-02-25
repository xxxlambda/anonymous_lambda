import re
from typing import Tuple, Any


def extract_code(text: str) -> tuple[bool, Any]:
    pattern = r'```python([^\n]*)(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches)>1:
        code_blocks = ''
        for match in matches:
            code_block = match[1]
            code_blocks += code_block
        return True, code_blocks
    elif len(matches):
        code_block = matches[-1]
        #if 'python' in code_block[0]:
        return True, code_block[1]
    else:
        return False, ''
