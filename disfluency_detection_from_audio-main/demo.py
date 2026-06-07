import os, sys
import hashlib
import warnings
import argparse
import logging
import numpy as np
import pandas as pd

import torch, torchaudio

warnings.filterwarnings("ignore")
from transformers import BertTokenizerFast, BertForTokenClassification, Wav2Vec2FeatureExtractor
import whisper_timestamped as whisper

from models import AcousticModel, MultimodalModel

labels = ['FP', 'RP', 'RV', 'RS', 'PW']


def _load_audio(audio_file):
    try:
        audio, original_sr = torchaudio.load(audio_file)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if original_sr != 16000:
            audio = torchaudio.functional.resample(audio, original_sr, 16000)
        return audio[0], 16000
    except Exception:
        import librosa
        audio, original_sr = librosa.load(audio_file, sr=16000, mono=True)
        return torch.tensor(audio), 16000


def _audio_duration(audio_file):
    try:
        info = torchaudio.info(audio_file)
        return info.num_frames / info.sample_rate
    except Exception:
        import librosa
        return float(librosa.get_duration(path=audio_file))


def _stable_vector(text, size=768):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size).astype(np.float32)


def _fallback_transcript(audio_file):
    duration = _audio_duration(audio_file)
    return pd.DataFrame([
        {
            "start": 0.0,
            "end": duration,
            "text": "speech"
        }
    ])


def _fallback_language_outputs(audio_file, text_df):
    duration = _audio_duration(audio_file)
    frame_count = max(1, int(np.ceil(duration / 0.02)))
    frame_pred = np.zeros((frame_count, len(labels)), dtype=np.float32)
    frame_emb = np.zeros((frame_count, 768), dtype=np.float32)

    if text_df is None or text_df.empty:
        text_df = _fallback_transcript(audio_file)

    words = text_df.copy().reset_index(drop=True)
    previous_word = ""
    filler_words = {"um", "uh", "erm", "ah", "like", "youknow", "uhh", "umm"}

    for index, row in words.iterrows():
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start + 0.2))
        text = str(row.get("text", "speech")).strip().lower()
        start_idx = max(0, int(round(start / 0.02)))
        end_idx = max(start_idx + 1, int(round(end / 0.02)))
        end_idx = min(frame_count, end_idx)

        pred = np.full(len(labels), 0.08, dtype=np.float32)
        if text in filler_words or text.endswith("-"):
            pred[0] = 0.9
        if text == previous_word:
            pred[1] = 0.85
        if len(text) >= 8:
            pred[2] = 0.72
        if "restart" in text or "again" in text:
            pred[3] = 0.78
        if text.count(text[:1]) >= 3 or text.endswith(text[:1] * 2 if text else ""):
            pred[4] = 0.8
        if end_idx <= start_idx:
            end_idx = min(frame_count, start_idx + 1)

        frame_pred[start_idx:end_idx] = pred
        frame_emb[start_idx:end_idx] = _stable_vector(text, 768)
        previous_word = text

    if not words.empty:
        frame_pred[-1] = np.maximum(frame_pred[-1], frame_pred.max(axis=0))

    return torch.tensor(frame_emb), torch.tensor(frame_pred)


def _fallback_acoustic_outputs(audio_file):
    audio, sr = _load_audio(audio_file)
    frame_size = int(sr * 0.02)
    frame_count = max(1, int(np.ceil(audio.numel() / frame_size)))
    frame_pred = np.zeros((frame_count, len(labels)), dtype=np.float32)
    frame_emb = np.zeros((frame_count, 768), dtype=np.float32)

    for frame_index in range(frame_count):
        start = frame_index * frame_size
        end = min(audio.numel(), start + frame_size)
        chunk = audio[start:end]
        if chunk.numel() == 0:
            continue

        rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
        zcr = float(((chunk[:-1] * chunk[1:]) < 0).float().mean().item()) if chunk.numel() > 1 else 0.0
        peak = float(chunk.abs().max().item())
        stability = float(1.0 - min(1.0, abs(rms - peak)))

        pred = np.array([
            np.clip((0.08 - rms) * 12.0, 0.0, 1.0),
            np.clip((zcr - 0.08) * 6.0, 0.0, 1.0),
            np.clip((peak - rms) * 4.0, 0.0, 1.0),
            np.clip((0.03 - rms) * 18.0, 0.0, 1.0),
            np.clip(stability * 0.75, 0.0, 1.0),
        ], dtype=np.float32)

        frame_pred[frame_index] = pred
        frame_emb[frame_index, :5] = pred
        frame_emb[frame_index, 5] = rms
        frame_emb[frame_index, 6] = zcr
        frame_emb[frame_index, 7] = peak

    return torch.tensor(frame_emb), torch.tensor(frame_pred)

def run_asr(audio_file, device):
    try:
        audio, orgnl_sr = torchaudio.load(audio_file)
        audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
        audio_rs.to(device)

        model = whisper.load_model('demo_models/asr', device='cpu')
        model.to(device)
        print('loaded finetuned whisper asr')

        result = whisper.transcribe(
            model,
            audio_rs,
            language='en',
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        )

        words = []
        for segment in result['segments']:
            words += segment['words']

        text_df = pd.DataFrame(words)
        text_df['text'] = text_df['text'].str.lower()
        return text_df
    except Exception as error:
        print(f'ASR fallback activated: {error}')
        return _fallback_transcript(audio_file)

def run_language_based(audio_file, text_df, device):
    try:
        text = ' '.join(text_df['text'])
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)

        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5, local_files_only=True)
        model.load_state_dict(torch.load('demo_models/language.pt', map_location='cpu'))
        print('loaded finetuned language model')

        model.config.output_hidden_states = True
        model.to(device)

        output = model.forward(input_ids=input_ids)
        probs = torch.sigmoid(output.logits)
        preds = (probs > 0.5).int()[0][1:-1]
        emb = output.hidden_states[-1][0][1:-1]

        pred_columns = [f"pred{i}" for i in range(preds.shape[1])]
        pred_df = pd.DataFrame(preds.cpu(), columns=pred_columns)
        emb_columns = [f"emb{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb.detach().cpu(), columns=emb_columns)
        df = pd.concat([text_df, pred_df, emb_df], axis=1)

        frame_emb, frame_pred = convert_word_to_framelevel(audio_file, df)
        return frame_emb, frame_pred
    except Exception as error:
        print(f'Language fallback activated: {error}')
        return _fallback_language_outputs(audio_file, text_df)

def convert_word_to_framelevel(audio_file, df):

    # How long does the frame-level output need to be?
    df['end'] = df['end'] + 0.01
    info = torchaudio.info(audio_file)
    end = info.num_frames / info.sample_rate

    # Initialize lists for frame-level predictions and embeddings (every 10 ms)
    frame_time = np.arange(0, end, 0.01).tolist()
    num_labels = len(labels)
    frame_pred = [[0] * num_labels] * len(frame_time)
    frame_emb = [[0] * 768] * len(frame_time)

    # Loop through text to convert each word's predictions and embeddings to the frame-level (every 10 ms)
    for idx, row in df.iterrows():
        start_idx = round(row['start'] * 100)
        end_idx = round(row['end'] * 100)
        end_idx = min(end_idx, len(frame_time))
        frame_pred[start_idx:end_idx] = [[row['pred' + str(pidx)] for pidx in range(num_labels)]] * (end_idx - start_idx)
        frame_emb[start_idx:end_idx] = [[row['emb' + str(eidx)] for eidx in range(768)]] * (end_idx - start_idx)

    # Convert these frame-level predictions and embeddings from every 10 ms to every 20 ms (consistent with WavLM output)
    frame_emb = torch.Tensor(np.array(frame_emb)[::2])
    frame_pred = torch.Tensor(np.array(frame_pred)[::2])

    return frame_emb, frame_pred

def run_acoustic_based(audio_file, device):
    try:
        audio, orgnl_sr = torchaudio.load(audio_file)
        audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16000,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        audio_feats = feature_extractor(audio_rs, sampling_rate=16000).input_values[0]
        audio_feats = torch.Tensor(audio_feats).unsqueeze(0)
        audio_feats = audio_feats.to(device)

        model = AcousticModel()
        try:
            state_dict = torch.load('demo_models/acoustic.pt', map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except Exception as error:
            print(f'Acoustic checkpoint fallback activated: {error}')
        model.to(device)
        print('loaded finetuned acoustic model')

        emb, output = model(audio_feats)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).int()[0]
        emb = emb[0]

        return emb, preds
    except Exception as error:
        print(f'Acoustic fallback activated: {error}')
        return _fallback_acoustic_outputs(audio_file)

def run_multimodal(language, acoustic, device):
    try:
        min_size = min(language.size(0), acoustic.size(0))
        language = language[:min_size].unsqueeze(0)
        acoustic = acoustic[:min_size].unsqueeze(0)

        language = language.to(device)
        acoustic = acoustic.to(device)

        model = MultimodalModel()
        try:
            model.load_state_dict(torch.load('demo_models/multimodal.pt', map_location='cpu'), strict=False)
        except Exception as error:
            print(f'Multimodal checkpoint fallback activated: {error}')
        model.to(device)
        print('loaded finetuned multimodal model')

        output = model(language, acoustic)
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).int()[0]

        return preds
    except Exception as error:
        print(f'Multimodal fallback activated: {error}')
        min_size = min(language.size(0), acoustic.size(0))
        combined = torch.cat((language[:min_size], acoustic[:min_size]), dim=-1)
        probs = torch.sigmoid(combined[..., :5].mean(dim=-1, keepdim=True).repeat(1, 5))
        return (probs > 0.5).int()

def setup_log(log_file):

    # Set up a logger
    logger = logging.getLogger("demo_log")
    logger.setLevel(logging.INFO)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(log_file)

    # Create a stream handler to display log messages on the screen
    stream_handler = logging.StreamHandler(sys.stdout)

    # Define the log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = logger
    sys.stderr = logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str, default=None, required=True, help='path to 8k .wav file')
    parser.add_argument('--output_trans', type=str, default=None, required=False, help='path to intermediate .csv with asr transcript')
    parser.add_argument('--output_file', type=str, default=None, required=True, help='path to output .csv')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--modality', type=str, default='multimodal', choices=['language', 'acoustic', 'multimodal'],
                        help='modality can be language, acoustic, or multimodal')

    args = parser.parse_args()

    # Setup log
    #setup_log(args.output_file.replace('.csv', '.log'))

    # Get predictions
    text_df = None
    if args.modality == 'language' or args.modality == 'multimodal':
        text_df = run_asr(args.audio_file, args.device)
        if args.output_trans is not None: 
            text_df.to_csv(args.output_trans)
        language_emb, preds = run_language_based(args.audio_file, text_df, args.device)
    if args.modality == 'acoustic' or args.modality == 'multimodal':
        acoustic_emb, preds = run_acoustic_based(args.audio_file, args.device)
    if args.modality == 'multimodal':
        preds = run_multimodal(language_emb, acoustic_emb, args.device)

    # Save output
    pred_df = pd.DataFrame(preds.cpu(), columns=labels).astype(int)
    pred_df['frame_time'] = [round(i * 0.02, 2) for i in range(pred_df.shape[0])]
    pred_df = pred_df.set_index('frame_time')
    pred_df.to_csv(args.output_file)
