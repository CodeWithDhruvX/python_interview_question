import whisper
import os
import subprocess

def format_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def transcribe_words_to_ass(video_path, language="en"):
    print("🔍 Loading Whisper model...")
    model = whisper.load_model("large")  # Use 'medium' or 'small' if GPU/CPU is limited

    print("🎙️ Transcribing with word-level timestamps...")
    result = model.transcribe(video_path, language=language, word_timestamps=True, verbose=True)

    ass_file = os.path.splitext(video_path)[0] + "_wordstyle.ass"
    with open(ass_file, "w", encoding="utf-8") as f:
        # ASS Header
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("Timer: 100.0000\n\n")

        # Styles
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, "
                "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: WordStyle,Arial,60,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,30,1\n\n")

        # Events
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        for segment in result["segments"]:
            for word in segment.get("words", []):
                start = format_ass_timestamp(word["start"])
                end = format_ass_timestamp(word["end"])
                text = word["word"].strip().replace('\n', '').replace('{', '').replace('}', '')
                if text:  # avoid empty words
                    f.write(f"Dialogue: 0,{start},{end},WordStyle,,0,0,0,,{text}\n")

    print(f"✅ ASS subtitle saved: {ass_file}")
    return ass_file

def burn_ass_subtitle(video_path, ass_path=None, output_path=None):
    if ass_path is None:
        ass_path = os.path.splitext(video_path)[0] + "_wordstyle.ass"
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_burned_wordstyle.mp4"

    print("🔥 Burning subtitles into video...")
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", video_path,
        "-vf", f"ass={ass_path}",
        "-c:a", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Final video with subtitles saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ FFmpeg error:", e)

# 🔹 Entry point
if __name__ == "__main__":
    video_file = "2025-07-18 21-54-56_220_output.mp4"

    # Step 1: Generate ASS subtitles with word-level sync
    ass_file = transcribe_words_to_ass(video_file, language="en")

    # Step 2: Burn subtitles into the video
    burn_ass_subtitle(video_file, ass_file)
