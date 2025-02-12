# [LAMBDA](https://www.polyu.edu.hk/ama/cmfai/lambda.html) - Multi-Agent Data Analysis System
<body>
<!-- <img src="https://github.com/user-attachments/assets/df454158-79e4-4da4-ae03-eb687fe02f16" style="width: 80%"> -->
<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/6f6d49ef-40b7-46f2-88ae-b8f6d9719c3a" style="width: 600px;">
</p> -->

![lambda_mix](https://github.com/user-attachments/assets/db5574aa-9441-4c9d-b44d-3b225d11e0cc)


We introduce **LAMBDA**, a novel open-source, code-free multi-agent data analysis system that harnesses the power of large models. LAMBDA is designed to address data analysis challenges in complex data-driven applications through the use of innovatively designed data agents that operate iteratively and generatively using natural language.

## Key Features

- **Code-Free Data Analysis**: Perform complex data analysis tasks through human language instruction.
- **Multi-Agent System**: Utilizes two key agent roles, the programmer and the inspector, to generate and debug code seamlessly.
- **User Interface**: This includes a robust user interface that allows direct user intervention in the operational loop.
- **Model Integration**: Flexibly integrates external models and algorithms to cater to customized data analysis needs.
- **Automatic Report Generation**: Concentrate on high-value tasks, rather than spending time and resources on report writing and formatting.

## Getting Started
### Installation
First, clone the repository.

```bash
git clone https://github.com/Stephen-SMJ/LAMBDA.git
cd LAMBDA
```

Then, we recommend creating a [Conda](https://docs.conda.io/en/latest/) environment for this project and install the dependencies by following commands:
```bash
conda create -n lambda python=3.10
conda activate lambda
```

Next, you should install the Jupyter kernel to create a local Code Interpreter:
```bash
ipython kernel install --name lambda --user
```

### Configuration
1. To use the Large Language Model, you should have an API key from [OpenAI](https://platform.openai.com/docs/guides/authentication) or other companies. Also, you can call your local LLMs once deployed by frameworks such as [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
2. **We used Aliyun Cloud Server to store the caches (like showing figures, models and so on). Currently, you should buy a [OSS（Object Storage Service](https://cn.aliyun.com/product/oss?from_alibabacloud=) from Aliyun to use it. But we will release a new version without the cloud server for easier use soon.**

3. Set your API key, models, working path, and OSS-related items in the config.yaml:
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
base_url_inspector : 'htts://api.openai.com/v1'
max_token_conv_model: 4096 # the max token of the conversation model, this will determine the maximum length of the report.


#================================================================================================
#                                       Config of the system
#================================================================================================
streaming : True

#cache_related
oss_endpoint: ""
oss_access_key_id: ""
oss_access_secret: ""
oss_bucket_name: ""
expired_time: 36000 # The expired time of the link in cache
cache_dir : "" # local cache dir
max_attempts : 5 # The max attempts of self-correcting
max_exe_time: 18000 # max time for the execution

#knowledge integration
retrieval : False # whether to start a knowledge retrieval. If you don't create your knowledge base, you should set it to False
mode : "full" # the mode of the #knowledge integration
```

Finally, Run the following command to start the demo with GUI:
```bash
python app.py
```


## Demonstration Videos

The performance of LAMBDA in solving data science problems is demonstrated in several case studies including:
- **[Data Analysis](https://www.polyu.edu.hk/ama/cmfai/files/lambda/lambda.mp4)**
- **[Integrating Human Intelligence](https://www.polyu.edu.hk/ama/cmfai/files/lambda/knw.mp4)**
- **[Education](https://www.polyu.edu.hk/ama/cmfai/files/lambda/LAMBDA_education.mp4)**

## Planning works

- Code refactoring.
- Remove the cloud cache module for easier use.
- Add a Docker image for easier use.
- Documentation writing.
- Replace Gradio with other GUI frameworks.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Acknowledgements

Thank the contributors and the communities for their support and feedback.

---

> If you find our work useful in your research, consider citing our paper by:



```bash
 @article{sun2024lambda,
          title={LAMBDA: A Large Model Based Data Agent},
          author={Sun, Maojun and Han, Ruijian and Jiang, Binyan and Qi, Houduo and Sun, Defeng and Yuan, Yancheng and Huang, Jian},
          journal={arXiv preprint arXiv:2407.17535},
          year={2024}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Stephen-SMJ/LAMBDA&type=Timeline)](https://star-history.com/#Stephen-SMJ/LAMBDA&Timeline)
</body>
