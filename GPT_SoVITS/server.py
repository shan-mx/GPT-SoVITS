import base64
import io
import os
import time

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from TTS_infer_pack.TTS import TTS

abs_path = "/".join(os.path.abspath(__file__).split("/")[:-1])

REF_AUDIO_DIR = f"{abs_path}/ref_audios"
if not os.path.exists(REF_AUDIO_DIR):
    os.makedirs(REF_AUDIO_DIR)

engine = TTS(
    {
        "version": "v2",
        "custom": {
            "device": "cuda",
            "is_half": True,
        },
    }
)

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def download_file(url: str, file_name: str):
    temp_filename = f"{file_name}.tmp"
    full_path = os.path.join(REF_AUDIO_DIR, file_name)
    temp_path = os.path.join(REF_AUDIO_DIR, temp_filename)

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            if response.status_code == 200:
                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                os.rename(temp_path, full_path)
                return True
    return False


class TTSRequest(BaseModel):
    text: str
    text_lang: str
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
    ref_audio_url: str
    ref_audio_id: str


@app.post("/tts")
async def tts(request: TTSRequest):
    ref_audio_file_suffix = request.ref_audio_url.split(".")[-1]
    ref_audio_file_name = f"{request.ref_audio_id}.{ref_audio_file_suffix}"
    ref_audio_path = os.path.join(REF_AUDIO_DIR, ref_audio_file_name)

    if not os.path.exists(ref_audio_path):
        success = await download_file(request.ref_audio_url, ref_audio_file_name)
        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to download reference audio"
            )

    start_time = time.time()
    request_data = request.model_dump()

    # fill in ref path for engine
    request_data["ref_audio_path"] = ref_audio_path
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
        # 16bit
        normalized_audio = ((i / np.max(np.abs(i))) * 32767).astype(np.int16)
        sf.write(wav, normalized_audio, sampling_rate, format="mp3")
        wav.seek(0)
        res.append(base64.b64encode(wav.read()))

    print(f"Time: {time.time() - start_time}")

    return res


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7865)
