This Repository was created in the context of the SciVQA shared task.  
## Folders
1. Finetuning
   This folder contains the code used to fine-tune the Qwen models. To use it you need to add the datasets and adapt the file paths accordingly.
2. Inference
   Here the code for the inference runs are located. First you start the model you want with ```finetunedModelInferenceServer.py```. Then you run the ```execution.py``` script.
3. Config
   The config contains the defaults.yaml file where you can set a variety of parameters (though some of which might get overwritten in some scripts). Most importantly are the dataset param which also sets the system prompt and the template_name. You can also enable few-shot prompting in the defaults.yaml.
4. OpenAI
   Here the scripts to perform runs on the OpenAI models are given
5. Dataset Filtering
   This folder contains the code to filter the ArXivQA and SpiQA datasets and to convert them to be combined with the SciVQA dataset.
6. Manual Evaluation
   This contains the code for the manual annotation of the generated answers.
7. Prompts
   This folder contains the prompt templates that were tested. The finally used ones are ```sys_prompt_11.txt``` as the system prompt and ```version_v7.j2``` in the zero-shot setting and ```version_v10.j2``` in the one-shot setting. In the one-shot setting ```few_shot_v1.j2``` fills the few shot variable field of the ```version_v10.j2``` template.
