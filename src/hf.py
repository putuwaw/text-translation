from huggingface_hub import hf_hub_download
import keras

def load_model_from_huggingface(repo_id: str, filename: str):
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        model = keras.saving.load_model(model_path)
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None
