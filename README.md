<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/banner.svg?raw=true" alt="DALL·E Flow: A Human-in-the-Loop workflow for creating HD images from text" width="60%">
<br>
<b>A Human-in-the-Loop<sup><a href="https://en.wikipedia.org/wiki/Human-in-the-loop">?</a></sup> workflow for creating HD images from text</b>
</p>

<p align=center>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.1k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
<a href="https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-orange?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>
<a href="https://hub.docker.com/r/jinaai/dalle-flow"><img alt="Docker Cloud Build Status" src="https://img.shields.io/docker/cloud/build/jinaai/dalle-flow?logo=docker&logoColor=white&style=flat-square"> <img alt="Docker Image Size (latest by date)" src="https://img.shields.io/docker/image-size/jinaai/dalle-flow?logo=docker&logoColor=white&style=flat-square"></a>

</p>




DALL·E Flow is an interactive workflow for generating high-definition images from text prompt. First, it leverages [DALL·E-Mega](https://github.com/borisdayma/dalle-mini) to generate image candidates, and then calls [CLIP-as-service](https://github.com/jina-ai/clip-as-service) to rank the candidates w.r.t. the prompt. The preferred candidate is fed to [GLID-3 XL](https://github.com/Jack000/glid-3-xl) for diffusion, which often enriches the texture and background. Finally, the candidate is upscaled to 1024x1024 via [SwinIR](https://github.com/JingyunLiang/SwinIR).

DALL·E Flow is built with [Jina](https://github.com/jina-ai/jina) in a client-server architecture, which gives it high scalability, non-blocking streaming, and a modern Pythonic interface. Client can interact with the server via gRPC/Websocket/HTTP with TLS.

**Why Human-in-the-Loop?** Generative art is a creative process. While recent advances of DALL·E unleash people's creativity, having a single-prompt-single-output UX/UI locks the imagination to a _single_ possibility, which is bad no matter how fine this single result is. DALL·E Flow is an alternative to the one-liner, by formalizing the generative art as an iterative procedure.

## Usage

DALL·E Flow is in client-server architecture.
- [Client usage](#Client)
- [Server usage, i.e. deploy your own server](#Server)


## Updates

- 🐳 **2022/6/21** [A prebuilt image is now available on Docker Hub!](https://hub.docker.com/r/jinaai/dalle-flow) This image can be run out-of-the-box on CUDA 11.6. Fix an upstream bug in CLIP-as-service. 
- ⚠️ **2022/5/23** Fix an upstream bug in CLIP-as-service. This bug makes the 2nd diffusion step irrelevant to the given texts. New Dockerfile proved to be reproducible on a AWS EC2 `p2.x8large` instance.
- **2022/5/13b** Removing TLS as Cloudflare gives 100s timeout, making DALLE Flow in usable [Please _reopen_ the notebook in Google Colab!](https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb).
- 🔐 **2022/5/13** New Mega checkpoint! All connections are now with TLS, [Please _reopen_ the notebook in Google Colab!](https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb).
- 🐳 **2022/5/10** [A Dockerfile is added! Now you can easily deploy your own DALL·E Flow](#run-in-docker). New Mega checkpoint! Smaller memory-footprint, the whole Flow can now fit into **one GPU with 21GB memory**.
- 🌟 **2022/5/7** New Mega checkpoint & multiple optimization on GLID3: less memory-footprint, use `ViT-L/14@336px` from CLIP-as-service, `steps 100->200`. 
- 🌟 **2022/5/6** DALL·E Flow just got updated! [Please _reopen_ the notebook in Google Colab!](https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb)
  - Revised the first step: 16 candidates are generated, 8 from DALL·E Mega, 8 from GLID3-XL; then ranked by CLIP-as-service.
  - Improved the flow efficiency: the overall speed, including diffusion and upscaling are much faster now!


## Gallery

<img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20realistic%20photo%20of%20a%20muddy%20dog.png?raw=true" width="32%" alt="a realistic photo of a muddy dog" title="a realistic photo of a muddy dog"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20scientist%20comparing%20apples%20and%20oranges%2C%20by%20Norman%20Rockwell.png?raw=true" width="32%" alt="A scientist comparing apples and oranges, by Norman Rockwell" title="A scientist comparing apples and oranges, by Norman Rockwell"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20portrait%20of%20the%20regal%20Burger%20King%20posing%20with%20a%20Whopper.png?raw=true" width="32%" alt="an oil painting portrait of the regal Burger King posing with a Whopper" title="an oil painting portrait of the regal Burger King posing with a Whopper"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/Eternal%20clock%20powered%20by%20a%20human%20cranium%2C%20artstation.png?raw=true" width="32%" alt="Eternal clock powered by a human cranium, artstation" title="Eternal clock powered by a human cranium, artstation"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/another%20planet%20amazing%20landscape.png?raw=true" width="32%" alt="another planet amazing landscape" title="another planet amazing landscape"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/The%20Decline%20and%20Fall%20of%20the%20Roman%20Empire%20board%20game%20kickstarter.png?raw=true" width="32%" alt="The Decline and Fall of the Roman Empire board game kickstarter" title="The Decline and Fall of the Roman Empire board game kickstarter"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20raccoon%20astronaut%20with%20the%20cosmos%20reflecting%20on%20the%20glass%20of%20his%20helmet%20dreaming%20of%20the%20stars%2C%20digital%20art.png?raw=true" width="32%" alt="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art" title="A raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars, digital art"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20photograph%20of%20an%20apple%20that%20is%20a%20disco%20ball%2C%2085%20mm%20lens%2C%20studio%20lighting.png?raw=true" width="32%" alt="A photograph of an apple that is a disco ball, 85 mm lens, studio lighting" title="A photograph of an apple that is a disco ball, 85 mm lens, studio lighting"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20cubism%20painting%20Donald%20trump%20happy%20cyberpunk.png?raw=true" width="32%" alt="a cubism painting Donald trump happy cyberpunk" title="a cubism painting Donald trump happy cyberpunk"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/oil%20painting%20of%20a%20hamster%20drinking%20tea%20outside.png?raw=true" width="32%" alt="oil painting of a hamster drinking tea outside" title="oil painting of a hamster drinking tea outside"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/Colossus%20of%20Rhodes%20by%20Max%20Ernst.png?raw=true" width="32%" alt="Colossus of Rhodes by Max Ernst" title="Colossus of Rhodes by Max Ernst"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/landscape%20with%20great%20castle%20in%20middle%20of%20forest.png?raw=true" width="32%" alt="landscape with great castle in middle of forest" title="landscape with great castle in middle of forest"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20medieval%20oil%20painting%20of%20Kanye%20west%20feels%20satisfied%20while%20playing%20chess%20in%20the%20style%20of%20Expressionism.png?raw=true" width="32%" alt="an medieval oil painting of Kanye west feels satisfied while playing chess in the style of Expressionism" title="an medieval oil painting of Kanye west feels satisfied while playing chess in the style of Expressionism"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/An%20oil%20pastel%20painting%20of%20an%20annoyed%20cat%20in%20a%20spaceship.png?raw=true" width="32%" alt="An oil pastel painting of an annoyed cat in a spaceship" title="An oil pastel painting of an annoyed cat in a spaceship"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/dinosaurs%20at%20the%20brink%20of%20a%20nuclear%20disaster.png?raw=true" width="32%" alt="dinosaurs at the brink of a nuclear disaster" title="dinosaurs at the brink of a nuclear disaster"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/fantasy%20landscape%20with%20medieval%20city.png?raw=true" width="32%" alt="fantasy landscape with medieval city" title="fantasy landscape with medieval city"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/GPU%20chip%20in%20the%20form%20of%20an%20avocado%2C%20digital%20art.png?raw=true" width="32%" alt="GPU chip in the form of an avocado, digital art" title="GPU chip in the form of an avocado, digital art"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20giant%20rubber%20duck%20in%20the%20ocean.png?raw=true" width="32%" alt="a giant rubber duck in the ocean" title="a giant rubber duck in the ocean"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/Paddington%20bear%20as%20austrian%20emperor%20in%20antique%20black%20%26%20white%20photography.png?raw=true" width="32%" alt="Paddington bear as austrian emperor in antique black & white photography" title="Paddington bear as austrian emperor in antique black & white photography"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20rainy%20night%20with%20a%20superhero%20perched%20above%20a%20city%2C%20in%20the%20style%20of%20a%20comic%20book.png?raw=true" width="32%" alt="a rainy night with a superhero perched above a city, in the style of a comic book" title="a rainy night with a superhero perched above a city, in the style of a comic book"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20synthwave%20style%20sunset%20above%20the%20reflecting%20water%20of%20the%20sea%2C%20digital%20art.png?raw=true" width="32%" alt="A synthwave style sunset above the reflecting water of the sea, digital art" title="A synthwave style sunset above the reflecting water of the sea, digital art"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20of%20ocean%20beach%20front%20in%20the%20style%20of%20Titian.png?raw=true" width="32%" alt="an oil painting of ocean beach front in the style of Titian" title="an oil painting of ocean beach front in the style of Titian"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20of%20Klingon%20general%20in%20the%20style%20of%20Rubens.png?raw=true" width="32%" alt="an oil painting of Klingon general in the style of Rubens" title="an oil painting of Klingon general in the style of Rubens"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/city%2C%20top%20view%2C%20cyberpunk%2C%20digital%20realistic%20art.png?raw=true" width="32%" alt="city, top view, cyberpunk, digital realistic art" title="city, top view, cyberpunk, digital realistic art"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20of%20a%20medieval%20cyborg%20automaton%20made%20of%20magic%20parts%20and%20old%20steampunk%20mechanics.png?raw=true" width="32%" alt="an oil painting of a medieval cyborg automaton made of magic parts and old steampunk mechanics" title="an oil painting of a medieval cyborg automaton made of magic parts and old steampunk mechanics"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20watercolour%20painting%20of%20a%20top%20view%20of%20a%20pirate%20ship%20sailing%20on%20the%20clouds.png?raw=true" width="32%" alt="a watercolour painting of a top view of a pirate ship sailing on the clouds" title="a watercolour painting of a top view of a pirate ship sailing on the clouds"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20knight%20made%20of%20beautiful%20flowers%20and%20fruits%20by%20Rachel%20ruysch%20in%20the%20style%20of%20Syd%20brak.png?raw=true" width="32%" alt="a knight made of beautiful flowers and fruits by Rachel ruysch in the style of Syd brak" title="a knight made of beautiful flowers and fruits by Rachel ruysch in the style of Syd brak"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%203D%20render%20of%20a%20rainbow%20colored%20hot%20air%20balloon%20flying%20above%20a%20reflective%20lake.png?raw=true" width="32%" alt="a 3D render of a rainbow colored hot air balloon flying above a reflective lake" title="a 3D render of a rainbow colored hot air balloon flying above a reflective lake"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20teddy%20bear%20on%20a%20skateboard%20in%20Times%20Square%20.png?raw=true" width="32%" alt="a teddy bear on a skateboard in Times Square " title="a teddy bear on a skateboard in Times Square "><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/cozy%20bedroom%20at%20night.png?raw=true" width="32%" alt="cozy bedroom at night" title="cozy bedroom at night"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20of%20monkey%20using%20computer.png?raw=true" width="32%" alt="an oil painting of monkey using computer" title="an oil painting of monkey using computer"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/the%20diagram%20of%20a%20search%20machine%20invented%20by%20Leonardo%20da%20Vinci.png?raw=true" width="32%" alt="the diagram of a search machine invented by Leonardo da Vinci" title="the diagram of a search machine invented by Leonardo da Vinci"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20stained%20glass%20window%20of%20toucans%20in%20outer%20space.png?raw=true" width="32%" alt="A stained glass window of toucans in outer space" title="A stained glass window of toucans in outer space"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20campfire%20in%20the%20woods%20at%20night%20with%20the%20milky-way%20galaxy%20in%20the%20sky.png?raw=true" width="32%" alt="a campfire in the woods at night with the milky-way galaxy in the sky" title="a campfire in the woods at night with the milky-way galaxy in the sky"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/Bionic%20killer%20robot%20made%20of%20AI%20scarab%20beetles.png?raw=true" width="32%" alt="Bionic killer robot made of AI scarab beetles" title="Bionic killer robot made of AI scarab beetles"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/The%20Hanging%20Gardens%20of%20Babylon%20in%20the%20middle%20of%20a%20city%2C%20in%20the%20style%20of%20Dal%C3%AD.png?raw=true" width="32%" alt="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí" title="The Hanging Gardens of Babylon in the middle of a city, in the style of Dalí"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/painting%20oil%20of%20Izhevsk.png?raw=true" width="32%" alt="painting oil of Izhevsk" title="painting oil of Izhevsk"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20hyper%20realistic%20photo%20of%20a%20marshmallow%20office%20chair.png?raw=true" width="32%" alt="a hyper realistic photo of a marshmallow office chair" title="a hyper realistic photo of a marshmallow office chair"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/fantasy%20landscape%20with%20city.png?raw=true" width="32%" alt="fantasy landscape with city" title="fantasy landscape with city"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/ocean%20beach%20front%20view%20in%20Van%20Gogh%20style.png?raw=true" width="32%" alt="ocean beach front view in Van Gogh style" title="ocean beach front view in Van Gogh style"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/An%20oil%20painting%20of%20a%20family%20reunited%20inside%20of%20an%20airport%2C%20digital%20art.png?raw=true" width="32%" alt="An oil painting of a family reunited inside of an airport, digital art" title="An oil painting of a family reunited inside of an airport, digital art"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/antique%20photo%20of%20a%20knight%20riding%20a%20T-Rex.png?raw=true" width="32%" alt="antique photo of a knight riding a T-Rex" title="antique photo of a knight riding a T-Rex"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20top%20view%20of%20a%20pirate%20ship%20sailing%20on%20the%20clouds.png?raw=true" width="32%" alt="a top view of a pirate ship sailing on the clouds" title="a top view of a pirate ship sailing on the clouds"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/an%20oil%20painting%20of%20a%20humanoid%20robot%20playing%20chess%20in%20the%20style%20of%20Matisse.png?raw=true" width="32%" alt="an oil painting of a humanoid robot playing chess in the style of Matisse" title="an oil painting of a humanoid robot playing chess in the style of Matisse"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20cubism%20painting%20of%20a%20cat%20dressed%20as%20French%20emperor%20Napoleon.png?raw=true" width="32%" alt="a cubism painting of a cat dressed as French emperor Napoleon" title="a cubism painting of a cat dressed as French emperor Napoleon"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/a%20husky%20dog%20wearing%20a%20hat%20with%20sunglasses.png?raw=true" width="32%" alt="a husky dog wearing a hat with sunglasses" title="a husky dog wearing a hat with sunglasses"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/A%20mystical%20castle%20appears%20between%20the%20clouds%20in%20the%20style%20of%20Vincent%20di%20Fate.png?raw=true" width="32%" alt="A mystical castle appears between the clouds in the style of Vincent di Fate" title="A mystical castle appears between the clouds in the style of Vincent di Fate"><img src="https://github.com/hanxiao/dalle/blob/gallery/.github/gallery/golden%20gucci%20airpods%20realistic%20photo.png?raw=true" width="32%" alt="golden gucci airpods realistic photo" title="golden gucci airpods realistic photo">

## Client

<a href="https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-orange?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>

Using client is super easy. The following steps are best run in [Jupyter notebook](./client.ipynb) or [Google Colab](https://colab.research.google.com/github/jina-ai/dalle-flow/blob/main/client.ipynb).  

You will need to install [DocArray](https://github.com/jina-ai/docarray) and [Jina](https://github.com/jina-ai/jina) first:

```bash
pip install "docarray[common]>=0.13.5" jina
```

We have provided a demo server for you to play:
> ⚠️ **Due to the massive requests, our server may be delay in response. Yet we are _very_ confident on keeping the uptime high.** You can also deploy your own server by [following the instruction here](#server).

```python
server_url = 'grpc://dalle-flow.jina.ai:51005'
```


### Step 1: Generate via DALL·E Mega

Now let's define the prompt:

```python
prompt = 'an oil painting of a humanoid robot playing chess in the style of Matisse'
```

Let's submit it to the server and visualize the results:

```python
from docarray import Document

da = Document(text=prompt).post(server_url, parameters={'num_images': 8}).matches

da.plot_image_sprites(fig_size=(10,10), show_index=True)
```

Here we generate 16 candidates, 8 from DALLE-mega and 8 from GLID3 XL, this is as defined in `num_images`, which takes about ~2 minutes. You can use a smaller value if it is too long for you. 


<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/client-dalle.png?raw=true" width="70%">
</p>

### Step 2: Select and refinement via GLID3 XL

The 16 candidates are sorted by [CLIP-as-service](https://github.com/jina-ai/clip-as-service), with index-`0` as the best candidate judged by CLIP. Of course, you may think differently. Notice the number in the top-left corner? Select the one you like the most and get a better view:

```python
fav_id = 3
fav = da[fav_id]
fav.display()
```

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/client-select1.png?raw=true" width="30%">
</p>

Now let's submit the selected candidates to the server for diffusion.

```python
diffused = fav.post(f'{server_url}', parameters={'skip_rate': 0.5, 'num_images': 36}, target_executor='diffusion').matches

diffused.plot_image_sprites(fig_size=(10,10), show_index=True)
```

This will give 36 images based on the selected image. You may allow the model to improvise more by giving `skip_rate` a near-zero value, or a near-one value to force its closeness to the given image. The whole procedure takes about ~2 minutes.

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/client-glid.png?raw=true" width="60%">
</p>

### Step 3: Select and upscale via SwinIR

Select the image you like the most, and give it a closer look:

```python
dfav_id = 34
fav = diffused[dfav_id]
fav.display()
```

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/client-select2.png?raw=true" width="30%">
</p>


Finally, submit to the server for the last step: upscaling to 1024 x 1024px.

```python
fav = fav.post(f'{server_url}/upscale')
fav.display()
```

That's it! It is _the one_. If not satisfied, please repeat the procedure.

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/client-select3.png?raw=true" width="50%">
</p>

Btw, DocArray is a powerful and easy-to-use data structure for unstructured data. It is super productive for data scientists who work in cross-/multi-modal domain. To learn more about DocArray, [please check out the docs](https://docs.jina.ai).

## Server

You can host your own server by following the instruction below.

### Hardware requirements

DALL·E Flow needs one GPU with 21GB VRAM at its peak. All services are squeezed into this one GPU, this includes (roughly)
- DALLE ~9GB
- GLID Diffusion ~6GB
- SwinIR ~3GB
- CLIP ViT-L/14-336px ~3GB

The following reasonable tricks can be used for further reducing VRAM:
- SwinIR can be moved to CPU (-3GB)
- CLIP can be delegated to [CLIP-as-service demo server](https://github.com/jina-ai/clip-as-service#text--image-embedding) (-3GB)


It requires at least 40GB free space on the hard drive, mostly for downloading pretrained models.

High-speed internet is required. Slow/unstable internet may throw frustrating timeout when downloading models.

CPU-only environment is not tested and likely won't work. Google Colab is likely throwing OOM hence also won't work.


### Server architecture

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/flow.svg?raw=true" width="70%">
</p>

If you have installed Jina, the above flowchart can be generated via:

```bash
# pip install jina
jina export flowchart flow.yml flow.svg
```

### Run in Docker

#### Prebuilt image

We have provided [a prebuilt Docker image](https://hub.docker.com/r/jinaai/dalle-flow) that can be pull directly.

```bash
docker pull jinaai/dalle-flow:latest
```

#### Build it yourself

We have provided [a Dockerfile](https://github.com/jina-ai/dalle-flow/blob/main/Dockerfile) which allows you to run a server out of the box.

Our Dockerfile is using CUDA 11.6 as the base image, you may want to adjust it according to your system.

```bash
git clone https://github.com/jina-ai/dalle-flow.git
cd dalle-flow

docker build --build-arg GROUP_ID=$(id -g ${USER}) --build-arg USER_ID=$(id -u ${USER}) -t jinaai/dalle-flow .
```

The building will take 10 minutes with average internet speed, which results in a 18GB Docker image.

#### Run container

To run it, simply do:

```bash
docker run -p 51005:51005 -v $HOME/.cache:/home/dalle/.cache --gpus all jinaai/dalle-flow
```

- The first run will take ~10 minutes with average internet speed.
- `-v $HOME/.cache:/root/.cache` avoids repeated model downloading on every docker run.
- The first part of `-p 51005:51005` is your host public port. Make sure people can access this port if you are serving publicly. The second par of it is [the port defined in flow.yml](https://github.com/jina-ai/dalle-flow/blob/e7e313522608668daeec1b7cd84afe56e5b19f1e/flow.yml#L4).

You should see the screen like following once running:

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/docker-run.png?raw=true" width="50%">
</p>

Note that unlike running natively, running inside Docker may give less vivid progressbar, color logs, and prints. This is due to the limitations of the terminal in a Docker container. It does not affect the actual usage.

### Run natively

Running natively requires some manual steps, but it is often easier to debug.

#### Clone repos

```bash
mkdir dalle && cd dalle
git clone https://github.com/jina-ai/dalle-flow.git
git clone https://github.com/JingyunLiang/SwinIR.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/hanxiao/glid-3-xl.git
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

There are couple models we need to download for GLID-3-XL:

```bash
cd glid-3-xl
wget https://dall-3.com/models/glid-3-xl/bert.pt
wget https://dall-3.com/models/glid-3-xl/kl-f8.pt
wget https://dall-3.com/models/glid-3-xl/finetune.pt
cd -
```

#### Install flow

```bash
cd dalle-flow
pip install -r requirements.txt
pip install jax==0.3.13
```

### Start the server

Now you are under `dalle-flow/`, run the following command: 

```bash
jina flow --uses flow.yml
```

You should see this screen immediately:

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/server-onstart.png?raw=true" width="50%">
</p>

On the first start it will take ~8 minutes for downloading the DALL·E mega model and other necessary models. The proceeding runs should only take ~1 minute to reach the success message.

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/server-wait.png?raw=true" width="50%">
</p>


When everything is ready, you will see:

<p align="center">
<img src="https://github.com/jina-ai/dalle-flow/blob/main/.github/server-success.png?raw=true" width="50%">
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
