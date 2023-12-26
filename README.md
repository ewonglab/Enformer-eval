# Instruction
1. First clone down the repository 
```bash
git clone git@github.com:ewonglab/Enformer-eval.git
```

2. Copy over over download the data folder
make sure you are in the clone repository
```bash
cp  -r /g/data/zk16/zelun/z_li_hon/wonglab_github/Enformer-eval/eval_script/data ./eval_script/data
```
or you could make a symlink to the data folder 
```bash
ln -s /g/data/zk16/zelun/z_li_hon/wonglab_github/Enformer-eval/eval_script/data ./eval_script/data
```
3. Need to reconfigure the paths in notebook for it to work

# Start a GPU node


1. Go to https://are.nci.org.au and follow the instruction in this google doc link
[are resource request example](https://docs.google.com/document/d/1nQpQeh6enuetnFB4gfyj_Fi1l2OVnp-irxOz89gtFVo/edit?usp=sharing)

2. to store the enformer data somewhere other than your home directory do this (https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory):
   
You can specify the cache directory everytime you load a model with .from_pretrained by the setting the parameter cache_dir. You can define a default location by exporting an environment variable TRANSFORMERS_CACHE everytime before you use (i.e. before importing it!) the library).

Example for python:

import os

os.environ['TRANSFORMERS_CACHE'] = '/blabla/cache/'

Example for bash:

export TRANSFORMERS_CACHE=/blabla/cache/
