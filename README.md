An interactive image to soundscape generation tool.

## Usage

```
python -m i2ss.cli run --image assets/test.jpg --queries "car.person.dog" --out out_vlm --vlm-device cuda:0
```

```
python -m i2ss.cli mix --tracks-dir out_test/tracks --meta-json out_test/tracks/meta.json --out out_test/mix/mix.wav
```

For Gradio app, close all proxy first

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
