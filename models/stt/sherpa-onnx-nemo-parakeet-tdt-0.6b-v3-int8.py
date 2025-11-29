import sherpa_onnx
import os
import model_utils


def load(cli_args):
    """see https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#

    Supports:
        Bulgarian (bg), Croatian (hr), Czech (cs), Danish (da), Dutch (nl)
        English (en), Estonian (et), Finnish (fi), French (fr), German (de)
        Greek (el), Hungarian (hu), Italian (it), Latvian (lv), Lithuanian (lt)
        Maltese (mt), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk)
        Slovenian (sl), Spanish (es), Swedish (sv), Russian (ru), Ukrainian (uk)

    """
    stt_model = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    stt_model_dir = os.environ.get("STT_MODEL_DIR", "/stt-models")

    model_utils.fetch_stt_model(stt_model_dir, stt_model)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(stt_model_dir, stt_model, "tokens.txt"),
        encoder=os.path.join(stt_model_dir, stt_model, "encoder.int8.onnx"),
        decoder=os.path.join(stt_model_dir, stt_model, "decoder.int8.onnx"),
        joiner=os.path.join(stt_model_dir, stt_model, "joiner.int8.onnx"),
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider=cli_args.provider,
        num_threads=cli_args.stt_thread_num,  # Adjust based on your hardware
        debug=cli_args.debug,
    )
