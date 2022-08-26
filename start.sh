#!/bin/sh
if test ${DISABLE_GLID3XL}; then
  echo "Latent diffusion checkpoints will not be downloaded because DISABLE_GLID3XL flag is on"
else
  if test -e /home/dalle/.cache/bert.pt -a -e /home/dalle/.cache/kl-f8.pt -a -e /home/dalle/.cache/finetune.pt; then
    echo "Latent diffusion checkpoints for glid3xl exist, continuing"
  else 
    echo "Latent diffusion checkpoints for glid3xl not exist, downloading"
    sudo apt update
    sudo apt install -y wget
    wget https://dall-3.com/models/glid-3-xl/bert.pt -O /home/dalle/.cache/bert.pt
    wget https://dall-3.com/models/glid-3-xl/kl-f8.pt -O /home/dalle/.cache/kl-f8.pt
    wget https://dall-3.com/models/glid-3-xl/finetune.pt -O /home/dalle/.cache/finetune.pt
  fi

  ln -s /home/dalle/.cache/bert.pt /dalle/glid-3-xl/bert.pt
  ln -s /home/dalle/.cache/kl-f8.pt /dalle/glid-3-xl/kl-f8.pt
  ln -s /home/dalle/.cache/finetune.pt /dalle/glid-3-xl/finetune.pt
fi

python3 flow_parser.py
jina flow --uses flow.tmp.yml