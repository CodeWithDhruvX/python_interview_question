
import whisper
import os
import subprocess
import json
import logging
import shlex
import platform
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")

def format_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def transcribe_words_to_ass(video_path, language="en"):
    logging.info("🔍 Loading Whisper model...")
    model = WhisperModel("large-v3", compute_type="int8")

    logging.info("🎙️ Transcribing with word-level timestamps...")
    segments, info = model.transcribe(video_path, language=language, word_timestamps=True, beam_size=5)

    ass_file = os.path.splitext(video_path)[0] + "_wordstyle.ass"
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write("""[Script Info]
Title: One Word at a Time Subs
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,24,&H00FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

""")
        for segment in segments:
            for word in segment.words:
                start = format_ass_timestamp(word.start)
                end = format_ass_timestamp(word.end)
                text = word.word.strip().replace('\n', '').replace('{', '').replace('}', '')
                if text:
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

    logging.info(f"✅ ASS subtitle saved: {ass_file}")
    return ass_file

def burn_ass_subtitle(video_path, ass_path, output_path):
    logging.info("🔥 Burning subtitles into video...")

    ass_path_ffmpeg = ass_path.replace("\\", "/")
    ass_path_ffmpeg = ass_path_ffmpeg.replace(":", "\\:")
    quoted_ass_path = shlex.quote(ass_path_ffmpeg)
    vf_string = f"ass={quoted_ass_path}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", vf_string,
        "-c:a", "copy",
        output_path
    ]

    logging.info(f"Executing FFmpeg command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅ Final video with subtitles saved: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg error: {e}")

def process_video_entry(entry):
    input_path = entry.get("input")
    output_path = entry.get("output")

    if not input_path or not os.path.exists(input_path):
        logging.error(f"❌ Input video file missing or invalid: {input_path}")
        return

    ass_file = transcribe_words_to_ass(input_path)
    burn_ass_subtitle(input_path, ass_file, output_path)

if __name__ == "__main__":
    json_file = "video_subtitle.json"
    if not os.path.exists(json_file):
        logging.error("❌ video_subtitle.json not found!")
    else:
        with open(json_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                process_video_entry(entry)
