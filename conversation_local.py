import os

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

    def run(self, conversation: dict):
        speakers = conversation["speakers"]
        defaults = conversation["defaults"]
        turns = conversation["turns"]

        audio_segments = []

        for i, turn in enumerate(turns):
            speaker = turn["speaker"]
            speaker_cfg = speakers[speaker]

            if not os.path.exists(speaker_cfg["voice_prompt"]):
                raise FileNotFoundError(f"Voice prompt not found: {speaker_cfg['voice_prompt']}")

            params = defaults.copy()
            params.update(turn.get("params", {}))

            wav = self.model.generate(
                turn["text"],
                audio_prompt_path=speaker_cfg["voice_prompt"],
                temperature=params["temperature"],
                min_p=params["min_p"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                exaggeration=params["exaggeration"],
                cfg_weight=params.get("cfg_weight", params.get("cfg", 0.5)),
            )

            audio_segments.append(wav)

            ta.save(
                os.path.join(self.output_dir, f"{i + 1:02d}_{speaker.lower()}.wav"),
                wav,
                self.sr,
            )

            pause_after = params.get("pause_after", 0)
            if i < len(turns) - 1 and pause_after > 0:
                silence = torch.zeros(1, int(pause_after * self.sr), device=wav.device)
                audio_segments.append(silence)

        combined = torch.cat(audio_segments, dim=1)
        ta.save(os.path.join(self.output_dir, "full_conversation.wav"), combined, self.sr)


if __name__ == "__main__":
    speakers = {
        "Samantha": {
            "voice_prompt": "data/samantha/combined.wav",
        },
        "Jesus": {
            "voice_prompt": "data/jesus/video_browser_use.wav",
        },
    }

    defaults = {
        "temperature": 0.8,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "pause_after": 0.5,
    }

    turns = [
        {
            "speaker": "Samantha",
            "text": "I was thinking about something strange today.",
        },
        {
            "speaker": "Jesus",
            "text": "Strange how?",
            "params": {
                "temperature": 0.9,
                "pause_after": 1.0,
            },
        },
        {
            "speaker": "Samantha",
            "text": "How breathing just happens without effort.",
            "params": {
                "exaggeration": 0.6,
            },
        },
        {
            "speaker": "Jesus",
            "text": "Maybe that's what it means to be carried by life.",
            "params": {
                "temperature": 0.65,
                "repetition_penalty": 1.3,
            },
        },
    ]

    conversation = {
        "speakers": speakers,
        "defaults": defaults,
        "turns": turns,
    }

    ConversationService().run(conversation)
