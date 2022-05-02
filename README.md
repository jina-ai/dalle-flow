# DALL·E Flow 

**Creating HD images from text via DALL·E in a Human-in-the-Loop workflow**

<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-2.8k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square" alt="Open in Google Colab"></a> <a href="https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-orange?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>

DALL·E Flow is a workflow for generating images from text prompt. It first leverages [DALL·E-Mega](https://github.com/borisdayma/dalle-mini) to generate image candidates, and then calls [CLIP-as-service](https://github.com/jina-ai/clip-as-service) to rank the candidates w.r.t. the prompt. The preferred candidate is fed to [GLID-3 XL](https://github.com/Jack000/glid-3-xl) for diffusion, which often enriches the texture and background. Finally, the candidate is upscaled to 1024x1024 with [SwinIR](https://github.com/JingyunLiang/SwinIR).

DALL·E Flow is built with [Jina]() in a client-server architecture. This enables DALL·E Flow with high scalability,  non-blocking streaming, and a modern Pythonic API. Client can interact the server via gRPC/Websocket/HTTP with TLS.

**Why Human-in-the-Loop?** Generative art is a creative process. While recent advances of DALL·E unleash people's creativity, having a single-prompt-single-output UX/UI locks the imagination _one_ possibility, which is bad no matter how fine this single result is. DALL·E Flow is an alternative to the one-liner, by formalizing the generative art as an iterative procedure.

## Gallery

> Image filename is the corresponding text prompt.

<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art.png?raw=true" width="32%" alt="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art" title="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/oil painting of a hamster drinking tea outside.png?raw=true" width="32%" alt="oil painting of a hamster drinking tea outside" title="oil painting of a hamster drinking tea outside">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/An oil pastel painting of an annoyed cat in a spaceship.png?raw=true" width="32%" alt="An oil pastel painting of an annoyed cat in a spaceship" title="An oil pastel painting of an annoyed cat in a spaceship">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/a rainy night with a superhero perched above a city, in the style of a comic book.png?raw=true" width="32%" alt="a rainy night with a superhero perched above a city, in the style of a comic book" title="a rainy night with a superhero perched above a city, in the style of a comic book">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/A synthwave style sunset above the reflecting water of the sea, digital art.png?raw=true" width="32%" alt="A synthwave style sunset above the reflecting water of the sea, digital art" title="A synthwave style sunset above the reflecting water of the sea, digital art">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/a 3D render of a rainbow colored hot air balloon flying above a reflective lake.png?raw=true" width="32%" alt="a 3D render of a rainbow colored hot air balloon flying above a reflective lake" title="a 3D render of a rainbow colored hot air balloon flying above a reflective lake">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/a teddy bear on a skateboard in Times Square .png?raw=true" width="32%" alt="a teddy bear on a skateboard in Times Square " title="a teddy bear on a skateboard in Times Square ">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/A stained glass window of toucans in outer space.png?raw=true" width="32%" alt="A stained glass window of toucans in outer space" title="A stained glass window of toucans in outer space">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/a campfire in the woods at night with the milky-way galaxy in the sky.png?raw=true" width="32%" alt="a campfire in the woods at night with the milky-way galaxy in the sky" title="a campfire in the woods at night with the milky-way galaxy in the sky">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí.png?raw=true" width="32%" alt="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí" title="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/An oil painting of a family reunited inside of an airport, digital art.png?raw=true" width="32%" alt="An oil painting of a family reunited inside of an airport, digital art" title="An oil painting of a family reunited inside of an airport, digital art">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/an oil painting of a humanoid robot playing chess in the style of Matisse.png?raw=true" width="32%" alt="an oil painting of a humanoid robot playing chess in the style of Matisse" title="an oil painting of a humanoid robot playing chess in the style of Matisse">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/.github/gallery/golden gucci airpods realistic photo.png?raw=true" width="32%" alt="golden gucci airpods realistic photo" title="golden gucci airpods realistic photo"> <img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/A stained glass window of toucans in outer space.png?raw=true" width="32%" alt="A stained glass window of toucans in outer space"> <img src="https://github.com/hanxiao/dalle/blob/main/.github/gallery/image (15).png?raw=true" width="32%">


## Client

<a href="https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-orange?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>

Using client is super easy. The following steps are best run in Jupyter notebook or [Google Colab]().  

The only dependency you will need is [DocArray](https://github.com/jina-ai/docarray).

```bash
pip install "docarray[common]>=0.13.5"
```

We have provided a demo server for you to play:

```python
server_url = 'grpc://dalle-flow.jina.ai:51005'
```


### Step 1: DALL·E Mega

Now let's define the prompt:

```python
prompt = 'an oil painting of a humanoid robot playing chess in the style of Matisse'
```

Let's submit it to the server and visualize the results:

```python
from docarray import Document

da = Document(text=prompt).post(server_url, parameters={'num_images': 16}).matches

da.plot_image_sprites(fig_size=(10,10), show_index=True)
```

Here we generate 16 candidates as defined in `num_images`, which takes about ~2 minutes. You can use a smaller value if it is too long for you. The results are sorted by [CLIP-as-service](https://github.com/jina-ai/clip-as-service), with index-`0` as the best candidate judged by CLIP. 


<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/client-dalle.png?raw=true" width="60%">
</p>

### Step 2: Select and refinement via GLID3 XL

Of course, you may think differently. Notice the number in the top-left corner? Select the one you like the most and get a better view:

```python
fav_id = 3
fav = da[fav_id]
fav.display()
```

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/client-select1.png?raw=true" width="30%">
</p>

Now let's submit the selected candidates to the server for diffusion.

```python
diffused = fav.post(f'{server_url}/diffuse', parameters={'skip_rate': 0.5}).matches

diffused.plot_image_sprites(fig_size=(10,10), show_index=True)
```

This will give 36 images based on the given image. You may allow the model to improvise more by giving `skip_rate` a near-zero value, or a near-one value to force its closeness to the given image. The whole procedure takes about ~2 minutes.

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/client-glid.png?raw=true" width="60%">
</p>

### Step 2: Select and upscale via SwanIR

Select the image you like the most, and give it a closer look:

```python
dfav_id = 34
fav = diffused[dfav_id]
fav.display()
```

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/client-select2.png?raw=true" width="30%">
</p>


Finally, submit to the server for the last step: upscaling to 1024 x 1024px.

```python
fav = fav.post(f'{server_url}/upscale')
fav.display()
```

That's it! It is _the one_. If not satisfied, please repeat the procedure.

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/client-select3.png?raw=true" width="50%">
</p>

Btw, DocArray is a powerful and easy-to-use data structure for unstructured data. It is super productive for data scientists who work in cross-/multi-modal domain. To learn more about DocArray, [please check out the docs](https://docs.jina.ai).

## Server

You can host your own server by following the instruction below.

### Hardware requirements

It is highly recommended to run DALL·E Flow on a GPU machine. In fact, one GPU is probably not enough. DALL·E Mega needs one with 22GB memory. SwinIR and GLID-3 also need one; as they can be spawned on-demandly in seconds, they can share one GPU.

It requires at least 40GB free space on the hard drive, mostly for downloading pretrained models.

CPU-only environment is not tested and likely won't work. Google Colab is likely throwing OOM hence also won't work.


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

On the first start it will take ~8 minutes for downloading the DALL·E mega model and other necessary models. The proceeding runs should only take ~1 minute to reach the success message.

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/server-wait.png?raw=true" width="50%">
</p>


When everything is ready, you will see:

<p align="center">
<img src="https://github.com/hanxiao/dalle/blob/main/.github/server-success.png?raw=true" width="50%">
</p>


Congrats! Now you should be able to [run the client](#client).

You can modify and extend the server flow as you like, e.g. changing the model, adding persistence, or even auto-posting to Instagram/OpenSea. With Jina and DocArray, you can easily make DALL·E Flow [cloud-native and ready for production](https://github.com/jina-ai/jina). 


<!-- start support-pitch -->
## Support

- To extend DALL·E Flow you will need to get familiar with  [Jina](https://github.com/jina-ai/jina) and [DocArray](https://github.com/jina-ai/docarray).
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