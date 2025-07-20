import os
import subprocess
import logging
import shlex
import json
import tkinter as tk
from tkinter import filedialog
from faster_whisper import WhisperModel
import ffmpeg

logging.basicConfig(level=logging.INFO, format="🔹 %(message)s")

def format_ass_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def get_video_duration(video_path):
    """Get video duration using ffprobe"""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except:
        logging.warning(f"⚠️ Could not get duration for {video_path}, using default 60s")
        return 60.0

def load_video_title_map():
    """Load video title mapping from JSON file"""
    try:
        with open("video_titles.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("⚠️ video_titles.json not found, using default titles")
        return []
    except json.JSONDecodeError:
        logging.warning("⚠️ Invalid JSON in video_titles.json, using default titles")
        return []

def get_title_for_video(input_video, video_title_map):
    """Get title text for a specific video"""
    for entry in video_title_map:
        if os.path.normpath(entry["slide_topic"]) == os.path.normpath(input_video):
            return entry["title_text"]
    return "Go Routines Simplified"  # Default title

def generate_title_overlay_ass(ass_path, video_duration, title_text):
    """Generate ASS file for title overlay"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("""[Script Info]
Title: Title Overlay
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleStyle,Impact,18,&H000000FF,&H00FFFF00,-1,0,0,0,100,100,0,0,1,3,0,8,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        # Show title for first 15 seconds or entire video duration if shorter
        title_duration = min(15, video_duration)
        f.write(f"Dialogue: 0,{format_time(0)},{format_time(title_duration)},TitleStyle,,0,0,0,,{title_text.upper()}\n")

def transcribe_words_to_ass(video_path, language="en"):
    logging.info("🔍 Loading Whisper model...")
    model = WhisperModel("large-v3", compute_type="int8")

    logging.info(f"🎙️ Transcribing: {video_path}")
    segments, _ = model.transcribe(video_path, language=language, word_timestamps=True, beam_size=5)

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

def combine_ass_files(word_ass_path, title_ass_path, combined_ass_path):
    """Combine word subtitles and title overlay into single ASS file"""
    logging.info("🔗 Combining subtitle files...")
    
    with open(word_ass_path, "r", encoding="utf-8") as f:
        word_content = f.read()
    
    with open(title_ass_path, "r", encoding="utf-8") as f:
        title_content = f.read()
    
    # Extract events section from title ASS
    title_events = []
    in_events = False
    for line in title_content.split('\n'):
        if line.strip().startswith('[Events]'):
            in_events = True
            continue
        elif line.strip().startswith('[') and in_events:
            break
        elif in_events and line.strip().startswith('Dialogue:'):
            title_events.append(line)
    
    # Combine with word subtitles
    with open(combined_ass_path, "w", encoding="utf-8") as f:
        f.write("""[Script Info]
Title: Combined Subtitles
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,24,&H00FFFFFF,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,90,1
Style: TitleStyle,Impact,18,&H000000FF,&H00FFFF00,-1,0,0,0,100,100,0,0,1,3,0,8,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

""")
        
        # Add title events first (higher priority)
        for event in title_events:
            f.write(event + '\n')
        
        # Add word subtitle events
        in_events = False
        for line in word_content.split('\n'):
            if line.strip().startswith('[Events]'):
                in_events = True
                continue
            elif line.strip().startswith('[') and in_events:
                break
            elif in_events and line.strip().startswith('Dialogue:'):
                f.write(line + '\n')

def burn_ass_subtitle(video_path, ass_path, output_path):
    logging.info("🔥 Burning subtitles into video...")

    ass_path_ffmpeg = ass_path.replace("\\", "/").replace(":", "\\:")
    vf_string = f"ass={shlex.quote(ass_path_ffmpeg)}"

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
        logging.info(f"✅ Subtitled video saved: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg error: {e}")

def convert_to_ts(video_path, output_ts):
    ffmpeg.input(video_path).output(output_ts, format="mpegts", vcodec="libx264", acodec="aac", strict="experimental").run(overwrite_output=True)

def merge_with_extra(main_video, extra_video, final_output):
    ts1 = "temp1.ts"
    ts2 = "temp2.ts"

    convert_to_ts(main_video, ts1)
    convert_to_ts(extra_video, ts2)

    ffmpeg.input(f"concat:{ts1}|{ts2}", format="mpegts").output(final_output, vcodec="copy", acodec="copy").run(overwrite_output=True)

    os.remove(ts1)
    os.remove(ts2)

def process_video_entry(video_path, output_dir, video_title_map, extra_video_path=None):
    if not video_path or not os.path.exists(video_path):
        logging.error(f"❌ Invalid input video: {video_path}")
        return

    # Get video duration and title
    video_duration = get_video_duration(video_path)
    title_text = get_title_for_video(video_path, video_title_map)
    logging.info(f"📺 Processing: {os.path.basename(video_path)} - Title: {title_text}")

    # Generate word-by-word subtitles
    word_ass_file = transcribe_words_to_ass(video_path)
    
    # Generate title overlay
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    title_ass_file = os.path.join(output_dir, f"{base_name}_title.ass")
    generate_title_overlay_ass(title_ass_file, video_duration, title_text)
    
    # Combine both subtitle files
    combined_ass_file = os.path.join(output_dir, f"{base_name}_combined.ass")
    combine_ass_files(word_ass_file, title_ass_file, combined_ass_file)

    # Generate output paths
    subtitled_path = os.path.join(output_dir, f"{base_name}_subtitled.mp4")
    final_output_path = os.path.join(output_dir, f"{base_name}_final.mp4")

    # Burn combined subtitles into video
    burn_ass_subtitle(video_path, combined_ass_file, subtitled_path)

    # Handle extra video merging
    if extra_video_path:
        merge_with_extra(subtitled_path, extra_video_path, final_output_path)
        os.remove(subtitled_path)
    else:
        os.rename(subtitled_path, final_output_path)

    # Cleanup temporary files
    try:
        os.remove(word_ass_file)
        os.remove(title_ass_file)
        os.remove(combined_ass_file)
        logging.info("🧹 Cleaned up temporary files")
    except OSError as e:
        logging.warning(f"⚠️ Could not clean up some temporary files: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # Load video title mapping
    video_title_map = load_video_title_map()
    logging.info(f"📋 Loaded {len(video_title_map)} video title mappings")

    input_videos = filedialog.askopenfilenames(title="Select Input Videos", filetypes=[("Video Files", "*.mp4 *.mov *.mkv")])
    if not input_videos:
        logging.warning("⚠️ No input videos selected.")
        exit()

    extra_video = filedialog.askopenfilename(title="Select Extra Video to Merge", filetypes=[("Video Files", "*.mp4 *.mov *.mkv")])
    if not extra_video:
        logging.warning("⚠️ No extra video selected.")
        extra_video = None

    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        logging.warning("⚠️ No output folder selected.")
        exit()

    for video_path in input_videos:
        process_video_entry(video_path, output_dir, video_title_map, extra_video)

    logging.info("🎉 All videos processed successfully!")