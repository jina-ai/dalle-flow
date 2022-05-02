# DALL·E Flow

DALL·E Flow is a workflow for generating images from text prompt. It first leverages DALL·E Mega to generate image candidates, then uses CLIP-as-service to rank those candidates w.r.t. the prompt. The preferred candidate is fed to [GLID-3 XL] for diffusion, which often  
GLID-3,

## Client

## Server

### Requirements

It is highly recommended to run DALL·E Flow on a GPU machine. In fact, one GPU is probably not enough. Besides DALL·E Mega, D 


### Install

#### Clone repos

```bash
mkdir dalle && cd dalle
git clone https://github.com/hanxiao/dalle-flow.git
git clone https://github.com/JingyunLiang/SwinIR.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/Jack000/glid-3-xl.git
```

You should have the following folder structure:

```text
dalle/
 |
 |-- dalle-flow/
 |-- SwinIR/
 |-- glid-3-xl/
 |-- latent-diffusion/
```

#### Install auxiliary repos

```bash
cd latent-diffusion && pip install -e . && cd -
cd glid-3-xl && pip install -e . && cd -
```

There are couple models we need to download first for GLID-3-XL:

```bash
wget https://dall-3.com/models/glid-3-xl/bert.pt
wget https://dall-3.com/models/glid-3-xl/kl-f8.pt
wget https://dall-3.com/models/glid-3-xl/finetune.pt
```

#### Install flow

```bash
cd dalle-flow
pip install -r requirements.txt
```

### Start the server

Now you are under `dalle-flow/`, run the following command: 

```bash
jina flow --uses flow.yml
```

You should see this screen immediately:

![](.github/server-onstart.png)

On the first start it will take ~8 minutes for downloading the DALL·E mega model and other necessary models.

![](.github/server-wait.png)

When everything is ready, you will see:

![](.github/server-success.png)

Congrats, now you should be able to [run the client](./client.ipynb).

The second run should only take ~1 minute to reach the address panel.