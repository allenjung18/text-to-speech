"""Render one passage in every en-US WaveNet and Standard voice via Google Cloud TTS.

Key comes from $GOOGLE_TTS_API_KEY (stored in the Keychain as google-tts-api-key). Prints the
characters billed so the free-tier spend stays visible.
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def api_key() -> str:
    # Python's own Keychain lookup comes back empty here, so the shell reads it:
    #   GOOGLE_TTS_API_KEY=$(security find-generic-password -a "$USER" -s google-tts-api-key -w) python3 audition.py
    key = os.environ.get("GOOGLE_TTS_API_KEY", "")
    if not key:
        sys.exit("GOOGLE_TTS_API_KEY is empty; read it from the Keychain first (see api_key)")
    return key


def synthesize(key: str, voice: str, text: str) -> bytes:
    body = {"input": {"text": text},
            "voice": {"languageCode": "en-US", "name": voice},
            "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"X-Goog-Api-Key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return base64.b64decode(json.load(resp)["audioContent"])


def main() -> None:
    # args: an optional passage .txt, then tiers (Wavenet, Standard) or full voice names
    args = sys.argv[1:]
    passage = HERE / (args.pop(0) if args and args[0].endswith(".txt") else "passage.txt")
    text = passage.read_text().strip()
    if len(text) > 4000:  # the API caps a request at 5,000 bytes; also keeps a bad passage from billing
        sys.exit(f"{passage.name} is {len(text):,} chars; expected a short audition passage")
    args = args or ["Wavenet", "Standard"]
    voices = [v for a in args for v in ([a] if "-" in a else [f"en-US-{a}-{c}" for c in "ABCDEFGHIJ"])]
    key = api_key()
    out = HERE / ("audition" if passage.name == "passage.txt" else f"audition-{passage.stem}")
    out.mkdir(exist_ok=True)
    for v in voices:
        (out / f"{v}.mp3").write_bytes(synthesize(key, v, text))
        print(f"  {v}")
    print(f"{len(voices)} voices x {len(text)} chars = {len(voices) * len(text):,} characters billed")


if __name__ == "__main__":
    main()
