# DALL·E Pipeline

## Server

### Install

It is highly recommended to run DALL·E Pipeline on a GPU machine.

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
git clone https://github.com/hanxiao/dalle-pipeline.git
cd dalle-pipeline
pip install -r requirements.txt
```

Now we can run the pipeline. In Jina's idiom a pipeline is a `Flow`:

```bash
jina flow --uses flow.yml
```

You should see this screen immedidately:

![](.github/server-onstart.png)

On the first start it will take ~8 minutes for downloading the DALL·E mega model and other necessary models.

![](.github/server-wait.png)

When everything is ready, you will see:

![](.github/server-success.png)

Congrats, now you should be able to [run the client](./client.ipynb).