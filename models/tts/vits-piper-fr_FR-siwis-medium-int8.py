import sherpa_onnx
import os
import model_utils


def load(cli_args):
    """fr_FR model with single voice
    """
    tts_model = "vits-piper-fr_FR-siwis-medium-int8"
    tts_model_dir = os.environ.get("TTS_MODEL_DIR", "/tts-models")

    model_utils.fetch_tts_model(tts_model_dir, tts_model)

    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(tts_model_dir, tts_model, "fr_FR-siwis-medium.onnx"),
                    lexicon="",
                    tokens=os.path.join(tts_model_dir, tts_model, "tokens.txt"),
                    data_dir=os.path.join(tts_model_dir, tts_model, "espeak-ng-data"),
                ),
                provider=cli_args.provider,
                num_threads=cli_args.tts_thread_num,
                debug=cli_args.debug,
            ),
            max_num_sentences=1,
        )
    )
