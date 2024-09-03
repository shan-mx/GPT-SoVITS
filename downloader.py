import os
import shutil
import subprocess

from huggingface_hub import snapshot_download


def download_gpt_sovits_models():
    print("Downloading GPT-SoVITS models...")
    model_dir = snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        local_dir="./GPT_SoVITS/pretrained_models",
        allow_patterns=[
            "gsv-v2final-pretrained/**",
            "chinese-hubert-base/**",
            "chinese-roberta-wwm-ext-large/**",
        ],
        use_auth_token=False,
    )
    print(f"GPT-SoVITS model downloaded to: {os.path.abspath(model_dir)}")


def download_g2pw_models():
    print("Downloading G2PW models...")
    try:
        # Download the G2PW model
        subprocess.run(
            [
                "wget",
                "-O",
                "G2PWModel_1.1.zip",
                "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip",
            ],
            check=True,
        )
        # Unzip the downloaded file
        subprocess.run(["unzip", "G2PWModel_1.1.zip"], check=True)
        # Move the unzipped directory to the desired location
        shutil.move("G2PWModel_1.1", "GPT_SoVITS/text/G2PWModel")
        # Remove the zip file
        os.remove("G2PWModel_1.1.zip")
        print("G2PW models downloaded and set up successfully.")
    except subprocess.CalledProcessError as error:
        print(f"An error occurred while downloading or extracting G2PW models: {error}")
    except OSError as error:
        print(f"An error occurred while moving or removing files: {error}")


if __name__ == "__main__":
    download_gpt_sovits_models()
    download_g2pw_models()
