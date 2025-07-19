import whisper
import os
import subprocess
import json
import logging
import shlex
import platform

logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")

def format_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def transcribe_words_to_ass(video_path, language="en"):
    logging.info("🔍 Loading Whisper model...")
    model = whisper.load_model("large")

    logging.info("🎙️ Transcribing with word-level timestamps...")
    result = model.transcribe(video_path, language=language, word_timestamps=True, verbose=True)

    ass_file = os.path.splitext(video_path)[0] + "_wordstyle.ass"
    with open(ass_file, "w", encoding="utf-8") as f:
        # Header and Styles using triple-quoted string
        f.write(f"""[Script Info]
Title: One Word at a Time Subs
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,60,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

""")

        for segment in result["segments"]:
            for word in segment.get("words", []):
                start = format_ass_timestamp(word["start"])
                end = format_ass_timestamp(word["end"])
                text = word["word"].strip().replace('\n', '').replace('{', '').replace('}', '')
                if text:
                    f.write(f"Dialogue: 0,{start},{end},WordStyle,,0,0,0,,{text}\n")

    logging.info(f"✅ ASS subtitle saved: {ass_file}")
    return ass_file

def burn_ass_subtitle(video_path, ass_path, output_path):
    logging.info("🔥 Burning subtitles into video...")

    # Ensure forward slashes for cross-platform compatibility
    ass_path_ffmpeg = ass_path.replace("\\", "/")

    # Escape colons for the ass filter (e.g., C:/ becomes C\:/)
    # This is crucial for FFmpeg to not misinterpret drive letters as filter options
    ass_path_ffmpeg = ass_path_ffmpeg.replace(":", "\\:")
    
    # Use shlex.quote to handle spaces and other special characters robustly
    # shlex.quote will add the necessary surrounding quotes if needed.
    quoted_ass_path = shlex.quote(ass_path_ffmpeg)
    
    # The vf_string should simply be 'ass=<quoted_path>'
    vf_string = f"ass={quoted_ass_path}"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", vf_string,
        "-c:a", "copy",
        output_path
    ]

    logging.info(f"Executing FFmpeg command: {' '.join(cmd)}") # Log the exact command
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

# 🔹 Entry point
if __name__ == "__main__":
    json_file = "video_subtitle.json"
    
    if not os.path.exists(json_file):
        logging.error("❌ video_titles.json not found!")
    else:
        with open(json_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
            for entry in entries:
                process_video_entry(entry)