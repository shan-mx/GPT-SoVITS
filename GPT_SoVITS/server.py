import base64
import io
import os
import time

import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块
from TTS_infer_pack.TTS import TTS

# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],
)  # 允许跨域的headers，可以用来鉴别来源等作用。


class TTSRequest(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    top_k: int
    top_p: float
    temperature: float
    text_split_method: str
    batch_size: int
    batch_threshold: float
    split_bucket: bool
    return_fragment: bool
    speed_factor: float


abs_path = "/".join(os.path.abspath(__file__).split("/")[:-1])

engine = TTS(
    {
        "custom": {
            "device": "cuda",
            "is_half": True,
            "t2s_weights_path": f"{abs_path}/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "vits_weights_path": f"{abs_path}/pretrained_models/s2G488k.pth",
            "cnhuhbert_base_path": f"{abs_path}/pretrained_models/chinese-hubert-base",
            "bert_base_path": f"{abs_path}/pretrained_models/chinese-roberta-wwm-ext-large",
            "flash_attn_enabled": True,
        }
    }
)


@app.post("/tts")
async def tts(request: TTSRequest):
    t = time.time()
    request_data = request.model_dump()

    request_data["ref_audio_path"] = (
        f"{abs_path}/ref_audios/{request_data['ref_audio_path']}"
    )
    gen = engine.run(request_data)

    try:
        sampling_rate, audio_sentences = next(gen)
    except StopIteration:
        raise HTTPException(
            status_code=400, detail="TTS engine failed to generate audio"
        )

    res = []
    for i in audio_sentences:
        wav = io.BytesIO()
        sf.write(wav, i, sampling_rate, format="wav")
        wav.seek(0)
        res.append(base64.b64encode(wav.read()))
    print(f"Time: {time.time() - t}")

    return res


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
