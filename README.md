# Auricula: Adaptive Unified Region-grounded Image-conditioned Composable User-controllable Layered Audio

An interactive image to soundscape generation tool.

## 1. Installation

1. Create conda environment

```
conda create -n i2ss python=3.10 -y
conda activate i2ss
```

2. Install torch

```
pip3 install torch torchvision
```

3. Install dependencies

```
pip install -U \
  diffusers transformers accelerate safetensors \
  numpy scipy librosa soundfile audioread \
  opencv-python pillow matplotlib tqdm einops \
  gradio typer pyyaml rich
```

4.  Install GroundingDino and SAM

```
mkdir -p third_party checkpoints

git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
git clone https://github.com/facebookresearch/segment-anything.git third_party/segment-anything

pip install -e third_party/segment-anything
pip install -e third_party/GroundingDINO
```

5.  Download weights

```
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

wget -O checkpoints/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

> You may have to choose different version of models to fit specific GPU specs.

6. Test installation

```
python -c "from groundingdino.util.inference import Model as GroundingDINOModel; print('ok')"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.__version__)"
```


## 2. Usage

### Specify `PYTHONPATH`

```
export PYTHONPATH=src
```

### Segmentation & Prompting

```
python -m i2ss.cli run --image <INPUT_IMAGE> --queries <QUERIES> --out <OUT_DIR> --vlm-device <DEVICE>

# Example
python -m i2ss.cli run --image assets/test.jpg --queries "car.person.dog" --out out_vlm --vlm-device cuda:0
```

### Mixing

```
python -m i2ss.cli mix --tracks-dir <TRACKS_DIR> --meta-json <META_JSON_FILE> --out <OUT_FILE>

# Example
python -m i2ss.cli mix --tracks-dir out_vlm/tracks --meta-json out_vlm/tracks/meta.json --out out_vlm/mix/mix.wav
```

### Gradio app

Ensure that proxies are closed

```
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
``` 

then

```
python ./demo/app.py
```

## Device recommendations

- Segmentation (GroundingDINO + SAM) accepts `--device` and `--sam_device` flags. Both default to `cpu` so the models stay on CPU unless you explicitly request CUDA.
- For RTX 4060 Ti 8GB, prefer `--device cpu --sam_device cpu` during segmentation; switch to GPU for the audio generation phase if available.
