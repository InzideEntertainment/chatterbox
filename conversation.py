import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


class ConversationService:
    def __init__(self, device=None, output_dir="output/conversation"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = self.model.sr
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_path(self, path: str, base_dir: Path) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = base_dir / p
        return str(p.resolve())

    def run(self, conversation: dict, base_dir: Path | None = None):
        base_dir = Path(base_dir) if base_dir is not None else Path(".")
        speakers = conversation["speakers"]
        defaults = conversation["defaults"]
        turns = conversation["turns"]

        required = ["temperature", "min_p", "top_p", "repetition_penalty", "exaggeration"]
        for key in required:
            if key not in defaults:
                raise ValueError(f"Missing default param: {key}")

        output_dir = conversation.get("output_dir", self.output_dir)
        output_dir = self._resolve_path(output_dir, base_dir)
        os.makedirs(output_dir, exist_ok=True)

        audio_segments = []

        for i, turn in enumerate(turns):
            speaker = turn["speaker"]
            speaker_cfg = speakers[speaker]

            voice_path = self._resolve_path(speaker_cfg["voice_prompt"], base_dir)
            if not os.path.exists(voice_path):
                raise FileNotFoundError(f"Voice prompt not found: {voice_path}")

            params = defaults.copy()
            params.update(turn.get("params", {}))

            wav = self.model.generate(
                turn["text"],
                audio_prompt_path=voice_path,
                temperature=params["temperature"],
                min_p=params["min_p"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                exaggeration=params["exaggeration"],
                cfg_weight=params.get("cfg_weight", params.get("cfg", 0.5)),
            )

            audio_segments.append(wav)

            ta.save(
                os.path.join(output_dir, f"{i + 1:02d}_{speaker.lower()}.wav"),
                wav,
                self.sr,
            )

            pause_after = params.get("pause_after", 0)
            if i < len(turns) - 1 and pause_after > 0:
                silence = torch.zeros(1, int(pause_after * self.sr), device=wav.device)
                audio_segments.append(silence)

        combined = torch.cat(audio_segments, dim=1)
        ta.save(os.path.join(output_dir, "full_conversation.wav"), combined, self.sr)

    def run_from_json(self, json_path: str):
        json_path = Path(json_path).resolve()
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        base_dir = payload.get("base_dir", json_path.parent)
        self.run(payload, base_dir=base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-speaker conversation from JSON.")
    parser.add_argument(
        "--conversation",
        "-c",
        type=str,
        required=True,
        help="Path to a JSON file with speakers/defaults/turns.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    args = parser.parse_args()

    service = ConversationService(device=args.device, output_dir=args.output_dir or "output/conversation")
    service.run_from_json(args.conversation)
