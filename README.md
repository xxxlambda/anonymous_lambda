# LAMBDA：A Large Model Based Data Agent (For Review)
---
> Note: This repository is only used for peer review and code review
<body>

We introduce **LAMBDA**, a novel open-source, code-free multi-agent data analysis system that harnesses the power of large models. LAMBDA is designed to address data analysis challenges in complex data-driven applications through the use of innovatively designed data agents that operate iteratively and generatively using natural language.

## Key Features

- **Code-Free Data Analysis**: Perform complex data analysis tasks through human language instruction.
- **Multi-Agent System**: Utilizes two key agent roles, the programmer and the inspector, to generate and debug code seamlessly.
- **User Interface**: This includes a robust user interface that allows direct user intervention in the operational loop.
- **Model Integration**: Flexibly integrates external models and algorithms to cater to customized data analysis needs.
- **Automatic Report Generation**: Concentrate on high-value tasks, rather than spending time and resources on report writing and formatting.
- **Jupyter Notebook Exporting**: Export the code and the results to Jupyter Notebook for reproduction and further analysis flexibly.

## Getting Started
### Installation
First, clone the repository.

```bash
git clone https://github.com/xxxlambda/anonymous_lambda.git
cd LAMBDA
```

Then, we recommend creating a [Conda](https://docs.conda.io/en/latest/) environment for this project and install the dependencies by following commands:
```bash
conda create -n lambda python=3.10
conda activate lambda
```

Then, install the required packages:
```bash
pip install -r requirements.txt
```

Next, you should install the Jupyter Kernel to create a local Code Interpreter:
```bash
ipython kernel install --name lambda --user
```

### Configuration to Easy Start
1. To use the Large Language Models, you should have an API key from [OpenAI](https://openai.com/api/pricing/) or other companies. Besides, we support OpenAI-Style interface for your local LLMs once deployed, available frameworks such as [LiteLLM](https://docs.litellm.ai/docs/), [ollama](https://ollama.com/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
> Here are some products that offer free APIkeys for your reference: [OpenRouter](https://openrouter.ai/) and [SILICONFLOW](https://siliconflow.cn/)
2. Set your API key, models and working path in the config.yaml:
```bash
#================================================================================================
#                                       Config of the LLMs
#================================================================================================
conv_model : "gpt-4o-mini" # the conversation model
programmer_model : "gpt-4o-mini"
inspector_model : "gpt-4o-mini"
api_key : ""
base_url_conv_model : 'https://api.openai.com/v1'
base_url_programmer : 'https://api.openai.com/v1'
base_url_inspector : 'https://api.openai.com/v1'


#================================================================================================
#                                       Config of the system
#================================================================================================
streaming : True
project_cache_path : "cache/conv_cache/" # local cache path
max_attempts : 5 # The max attempts of self-correcting
max_exe_time: 18000 # max time for the execution

#knowledge integration
retrieval : False # whether to start a knowledge retrieval. If you don't create your knowledge base, you should set it to False
```

Finally, Run the following command to start the LAMBDA with GUI:
```bash
python app.py
```


## Demonstration Videos

The performance of LAMBDA in solving data science problems is demonstrated in several case studies including:
- **[Data Analysis](https://www.youtube.com/watch?v=fGvXWWeUH8A)**
- **[Integrating Human Intelligence](https://www.youtube.com/watch?v=sfU2ZzvNke0)**
- **[Education](https://www.youtube.com/watch?v=8q-EFoak9ZI)**

