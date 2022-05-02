# DALL·E Flow 

**Creating HD images from text via DALL·E in a Human-in-the-Loop workflow**

<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-2.8k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>



DALL·E Flow is a workflow for generating images from text prompt. It first leverages [DALL·E-Mega](https://github.com/borisdayma/dalle-mini) to generate image candidates, and then calls [CLIP-as-service](https://github.com/jina-ai/clip-as-service) to rank the candidates w.r.t. the prompt. The preferred candidate is fed to [GLID-3 XL](https://github.com/Jack000/glid-3-xl) for diffusion, which often enriches the texture and background. Finally, the candidate is upscaled to 1024x1024 with [SwinIR](https://github.com/JingyunLiang/SwinIR).

DALL·E Flow is built with [Jina]() in a client-server architecture. This enables DALL·E Flow with high scalability,  non-blocking streaming, and a modern Pythonic API. Client can interact the server via gRPC/Websocket/HTTP with TLS.

## Gallery

> Image filename (dash is whitespace) is the corresponding text prompt.

## Client

Using client is super easy. The following steps are best run in Jupyter notebook or [Google Colab]().  

The only dependency you will need is [DocArray](https://github.com/jina-ai/docarray).

```bash
pip install "docarray[common]>=0.13.5"
```

We have provided a demo server for you to play:

```python
server_url = 'grpc://dalle-flow.jina.ai:51005'
```

Now let's define the prompt:

```python
prompt = ''
```

Let's submit it to the server and visualize the results:

```python
from docarray import Document

da = Document(text=prompt).post(server_url, parameters={'num_images': 16}).matches

da.plot_image_sprites(fig_size=(10,10), show_index=True)
```

Here we generate 16 candidates as defined in `num_images`, which takes about ~2 minutes. You can use a smaller value if it is too long for you. The results are sorted by [CLIP-as-service](https://github.com/jina-ai/clip-as-service), with index-`0` as the best candidate judged by CLIP. 

Of course, you may think differently. So select the one you like the most and zoom in to get better view:

```python
fav_id = 14
fav = da[fav_id]
fav.display()
```

Now let's submit the selected candidates to the server for diffusion.

```python
diffused = fav.post(f'{server_url}/diffuse', parameters={'skip_rate': 0.5}).matches

diffused.plot_image_sprites(fig_size=(10,10), show_index=True)
```

This will give 36 images based on the given image. You may allow the model to improvise more by giving `skip_rate` a near-zero value, or a near-one value to force its closeness to the given image. The whole procedure takes about ~2 minutes.

Select the image you like the most, and then submit to the server for the final step: upscaling to 1024 x 1024px.

```python
fav = fav.post(f'{server_url}/upscale')
fav.display()
```

That's it! It is _the one_. If not satisfied, please repeat the procedure.

To learn more about DocArray API, [please follow the docs](https://docs.jina.ai).

## Server

You can host your own server by following the instruction below.

### Hardware requirements

It is highly recommended to run DALL·E Flow on a GPU machine. In fact, one GPU is probably not enough. DALL·E Mega needs one with 22GB memory. SwinIR and GLID-3 also need one; as they can be spawned on-demandly in seconds, they can share one GPU.

It requires at least 40GB free space on the hard drive, mostly for downloading pretrained models.

CPU-only environment is not tested and likely won't work.


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

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/server-onstart.png?raw=true" width="50%">
</p>

On the first start it will take ~8 minutes for downloading the DALL·E mega model and other necessary models.

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/server-wait.png?raw=true" width="50%">
</p>


When everything is ready, you will see:

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/server-success.png?raw=true" width="50%">
</p>


Congrats, now you should be able to [run the client](./client.ipynb).

The second run should only take ~1 minute to reach the address panel.

<!-- start support-pitch -->
## Support

- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Join our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) meet-up to discuss your use case and learn Jina's new features.
    - **When?** The second Tuesday of every month
    - **Where?**
      Zoom ([see our public events calendar](https://calendar.google.com/calendar/embed?src=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&ctz=Europe%2FBerlin)/[.ical](https://calendar.google.com/calendar/ical/c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com/public/basic.ics))
      and [live stream on YouTube](https://youtube.com/c/jina-ai)
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## Join Us

DALL·E Flow is backed by [Jina AI](https://jina.ai) and licensed under [Apache-2.0](./LICENSE). [We are actively hiring](https://jobs.jina.ai) AI engineers, solution engineers to build the next neural search ecosystem in open-source.

<!-- end support-pitch -->