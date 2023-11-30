import os
import sys
import traceback
import tqdm
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6]
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7]
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(sys.argv)
model_path = "assets/hubert/hubert_base.pt"

printt(exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    #feats = torch.from_numpy(wav).float()
    feats = torch.from_numpy(wav)
    if is_half:
        feats = feats.half()
    else:
        feats = feats.float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
os.system('cls' if os.name == 'nt' else 'clear')
print("Starting feature extraction...\n")
printt("Loaded model {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt("Using %s" % device)
if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    os.system('cls' if os.name == 'nt' else 'clear')
    printt("An error occurred in the feature extraction, make sure you have provided the audios correctly.")
else:
    printt("- %s" % len(todo))
    with tqdm.tqdm(total=len(todo)) as pbar:
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": feats.to(device),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0]) if version == "v1" else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        printt("%s-contains nan" % file)
                    # if idx % n == 0:
                    #     printt("now-%s,all-%s,%s,%s" % (idx, len(todo), file, feats.shape))
                    pbar.set_description(f"Processing: %s - Shape: %s" % (file, feats.shape))
            except:
                printt(traceback.format_exc())
            pbar.update(1)
    printt("\nFeature extraction completed successfully!")
