import logging
import os

# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
import gradio as gr
from dotenv import load_dotenv

from assets.configs.config import Config
from assets.i18n.i18n import I18nAuto
from lib.infer.modules.vc.pipeline import Pipeline
VC = Pipeline

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
#(i18n)

load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


app = gr.Blocks()
with app:
    with gr.Tabs():
