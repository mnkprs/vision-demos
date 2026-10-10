"""Params for the deadlift run: pose, plate tracking, reps and back position.

Edit values here, then run `python main.py`.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# A clip, or a folder: every video directly inside it runs in turn, and
# subfolders (data/input/archive/, say) are left alone. Clips or folders named on
# the command line replace it for that run; `python main.py -h` lists the flags.
INPUT = DATA_DIR / "input"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted: ffmpeg applies the source rotation, OpenCV ignores it. Two
# renditions, because they answer to different limits: the models see the
# inference one (ViTPose caps at a 2048px long edge), while the export answers
# only to what you want to watch. Every model's output is in normalized
# coordinates, so one set of answers renders at any export size, and changing
# EXPORT_HEIGHT costs no new gateway call.
INFERENCE_HEIGHT = 1080   # what the models see; changing it re-runs every model
EXPORT_HEIGHT = 1080      # the rendered video; None keeps the source resolution
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# ── HDR ──────────────────────────────────────────────────────────────────────
# iPhone HDR clips are HLG Rec. 2020; OpenCV reads them as sRGB, so they look
# washed out. "auto" tone-maps only HDR-tagged sources, "off" never, "force" always.
# Applies to the exported video only.
TONEMAP = "auto"

# `bt.2446a` is the safe default; `spline` keeps midtones brighter, `hable` is filmic.
TONEMAP_ALGORITHM = "bt.2446a"

# True also feeds the models tone-mapped frames (re-runs every model).
TONEMAP_INFERENCE = False

# ── Gateway ──────────────────────────────────────────────────────────────────
OPENAI_BASE_URL = "https://gateway.vlm.run/v1/openai"    # ViTPose, SAM 3.1
TYPESAFE_BASE_URL = "https://gateway.vlm.run/typesafe"   # System One
VIDEO_REQUEST_TIMEOUT = 1800.0   # seconds; a video call is minutes, not seconds

# Every gateway call is cached on its inputs. True reuses a matching cached reply
# instead of calling again; a changed setting is always a miss.
REUSE_POSES = True
REUSE_PLATE = True
REUSE_READS = True

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
EVERY_FRAME = True       # detect on every frame; costs wall clock, not money
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)
MAX_PERSONS = 1          # one lifter
PERSON_IOU_THRESHOLD = 0.6

DRAW_FACE = False
# Which joints the skeleton keeps. Only edges between two kept joints are drawn,
# on both sides: shoulder-hip and hip-knee, which is the hinge. None draws the
# full skeleton.
DRAW_JOINTS = ("shoulder", "hip", "knee")
LINE_THICKNESS = 3       # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4

# ── Plate: SAM 3.1 ───────────────────────────────────────────────────────────
PLATE_MODEL = "facebook/sam3.1"
PLATE_PROMPT = "barbell weight plate"
# SAM's tracker keeps at most 128 samples per call. A stride of 5 covers 640
# frames (21 s at 30 fps) in one call, a look every 0.17 s; a longer clip is cut
# into segments rather than sampled more sparsely.
PLATE_TRACK_STRIDE = 5
PLATE_TRACK_MAX_FRAMES = 128
PLATE_MIN_SCORE = 0.5
PLATE_MIN_AREA_FRACTION = 0.4   # of the typical plate area; drops small false positives
PLATE_SIZE_TOLERANCE = 0.25     # a box whose height is off the plate's by more is not the plate
# The ruler for bar height. ASSUMPTION: an Olympic plate, 450 mm across (IWF
# standard for every bumper plate and for 20/25 kg iron). Change this for other
# plates; every centimeter in the run scales with it. The plate's *vertical*
# diameter is used, which a camera panned off side-on does not shorten.
PLATE_DIAMETER_CM = 45.0

PLATE_COLOR = (255, 160, 40)    # BGR, blue: outline and bar path
PLATE_FILL_OPACITY = 0.25
# A fading comet behind the plate's center, in the plate's color, drawn with
# running's comet compositor: the width tapers and the alpha fades from the
# head back. Same look as running's ankle trails; no wind, since the bar moves
# straight up and down.
BAR_PATH_SECONDS = 0.8          # trail length behind the plate; 0 disables it
# A lighter tint of PLATE_COLOR: the same blue, but it has to read over the
# plate's own blue fill, which the trail runs straight through.
BAR_TRAIL_COLOR = (255, 215, 150)   # BGR
BAR_TRAIL_WIDTH = 9             # px at a 720px-wide frame, at the head
BAR_TRAIL_TAPER = 0.18          # tail width as a fraction of the head's
BAR_TRAIL_OPACITY = 0.85        # alpha at the head; the tail fades to nothing
BAR_TRAIL_FADE = 1.35
BAR_TRAIL_GLOW = 2.6            # halo width, as a multiple of the core's
BAR_TRAIL_GLOW_OPACITY = 0.28
BAR_TRAIL_SOFTNESS = 0.6        # edge blur, as a fraction of the head width

# ── Reps ─────────────────────────────────────────────────────────────────────
# Counted on bar height, the same hysteresis as chin_ups: armed on the floor,
# confirmed past REP_PEAK_FRACTION of the typical rep, closed under RESET.
SMOOTH_MEAN_FRAMES = 5        # centered moving average on the interpolated height
BASELINE_PERCENTILE = 10      # the floor: where the plate spends most of the clip
REP_PEAK_FRACTION = 0.60
REP_RESET_FRACTION = 0.30
REP_FLOOR_FRACTION = 0.08     # under this share of rep height, the plate is down
REP_ONSET_CM = 2.0            # the bar has left the floor
# Lockout is the first frame at this share of the rep's top, not the top itself:
# the bar holds there, and the highest frame can land anywhere in the hold. At
# 0.98 the hip is within ~2 degrees of fully open on both test clips.
REP_LOCKOUT_FRACTION = 0.98
REP_MIN_HEIGHT_CM = 10.0      # below this, nothing in the clip is a rep

# Which signal the reps come from. "auto" takes the plate and falls back to the
# hip hinge when SAM fails, finds no plate, or finds no rep in it; "plate" and
# "hinge" force one ("hinge" also skips the SAM call). The hinge is computed and
# cross-checked against the plate on every run either way.
REP_SOURCE = "auto"

# ── Hip hinge: ViTPose ───────────────────────────────────────────────────────
# The angle at the hip between shoulder and knee. Bent over the bar the three
# make a triangle; at lockout it collapses into a line near 180 degrees. Counted
# with the same hysteresis fractions as the bar, on the angle's extension above
# the bottom of the clip.
HINGE_MIN_SCORE = 0.3         # ViTPose confidence needed on shoulder, hip and knee
HINGE_SMOOTH_FRAMES = 5       # median, then mean, over this many frames
HINGE_ONSET_FRACTION = 0.10   # the pull starts once the hip has opened this share of a rep
HINGE_MIN_DEGREES = 30.0      # below this much extension, nothing in the clip is a rep
HINGE_MATCH_OVERLAP = 0.5     # plate and hinge reps whose spans overlap this much are one rep

# The hinge drawn on the lifter: an arc between the skeleton's shoulder-hip and
# hip-knee lines, and the angle's live value. Shows the fallback signal even when the
# reps come from the plate.
DRAW_HIP_ANGLE = True
HINGE_COLOR = (255, 255, 255)   # BGR: the arc and its value
HINGE_ARC_RADIUS = 26           # px at a 720px-wide frame; scales with the export
HINGE_LABEL_SIZE = 22
# The number beside the arc: "value" (139°), "hip" (hip 139°), or "none" (arc
# only). The hip angle segments reps, it does not judge the back; "hip" says so.
HINGE_LABEL = "none"

# ── Back position: TypeSafe System One ───────────────────────────────────────
# Any System One model works here (`GET /typesafe/v1/models`). Gemma-4 was the
# one that told a flat back leaning forward from a rounded one, across both
# test clips; DiffusionGemma, the route's default, read lean as rounding.
READ_MODEL = "google/gemma-4-26b-a4b-it"
READ_TIMEOUT = 60.0      # seconds, per read; one read is a single constrained pass
MAX_RETRIES = 2          # per read, on 429/5xx
CONCURRENCY = 8          # reads in flight at once

# One read per sampled frame; frames in between are interpolated. Every frame
# by default: a fast pull is off the floor and past the knee in 0.3 s, and at
# 10 reads/s its verdict would rest on three reads.
SAMPLE_FPS = None        # reads per second of video; None reads every frame
IMAGE_LONG_EDGE = 1024   # px; the "high" vision budget is 280 tokens regardless
IMAGE_DETAIL = "high"    # "high" (280 tokens) or "low" (70, 4x cheaper)
JPEG_QUALITY = 90

# Crop each read to the ViTPose person box, padded by this share of the box on
# every side. The vision budget then goes on the lifter, not the gym.
CROP_TO_PERSON = True
PERSON_CROP_PAD = 0.15
CROP = None    # a fixed normalized (x0, y0, x1, y1), used when CROP_TO_PERSON is False

STATE = "A side view of a person performing a barbell deadlift."
# About the spine's shape, not its angle: every deadlift starts leaning far
# forward, and a question that does not say so gets lean back as "rounded".
INSTRUCTIONS = (
    "Judge the shape of the lifter's spine, not how far the torso leans forward. "
    "Is the back straight (neutral) or rounded (flexed)?"
)
LABELS = {
    "straight": "Neutral spine: the back is flat from hips to neck, chest up, head in line "
                "with the torso. A flat back can lean far forward and is still straight.",
    "rounded": "Flexed spine: the upper back bows outward into a hump, the shoulders roll "
               "forward and the head drops toward the floor.",
}
POSITIVE_LABEL = "rounded"   # the probability the panel plots
THRESHOLD = 0.5              # above this, a frame or a rep is called POSITIVE_LABEL

SMOOTH_READS = 5        # odd median window over reads, to kill single-read flips; 1 disables
INTERPOLATE = "linear"  # between reads: "linear" or "hold"

# A rep's verdict is the mean P(rounded) over its first pull: from the bar
# leaving the floor until it reaches this share of the rep's height (about the
# knee). Rounding shows there; standing tall, every back reads straight, and a
# crouched setup reads rounded even with a flat back.
BACK_WINDOW_FRACTION = 0.5
# The verdict averages the per-frame curve after a Gaussian with this sigma, in
# seconds: enough to damp a burst of flipping reads, short enough not to drag in
# the setup or the lockout around a window that is often under half a second.
# Tighter than the panel's (PANEL_P_SMOOTH_SECONDS), which is chosen for looks;
# a run warns if the two ever disagree about a rep.
BACK_SMOOTH_SECONDS = 0.15

# ── Side panel ───────────────────────────────────────────────────────────────
# The clip on the left, an equal-width stats panel on the right. False writes
# the overlaid clip alone, at its own size (skeleton, hinge arc, plate, bar
# path and the read box, nothing beside it); timeline.png and the JSON files
# are unchanged.
SIDE_PANEL = True
PANEL_HEADER_SIZE = 44     # px at a 720px-wide panel; scales with the output
PANEL_VERDICT_SIZE = 34    # the "Back: ..." line; shrinks further if it would meet the %
PANEL_ROW_SIZE = 26        # the per-rep rows
PANEL_NOTE_SIZE = 20
# Display smoothing of the P(rounded) graph, live label and box color: a
# Gaussian with this sigma, in seconds. Looks only; verdicts use
# BACK_SMOOTH_SECONDS, and the raw reads in reads.json are saved unsmoothed.
# 0 disables it.
PANEL_P_SMOOTH_SECONDS = 0.3
PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

# The credit, bottom-right of the panel: one entry per line, top to bottom. Each
# entry is either ("LABEL", "value"), drawn as "LABEL: value" with the label a
# shade dimmer, or a plain string drawn as it is. Same format as rock_climbing_3d:
#
#   CREDIT = ["Jeremy Park"]                        # a name alone
#   CREDIT = [("X", "@jeremyparkphd")]              # a handle alone
#   CREDIT = [("LI", "Jeremy Park, PhD"), ("X", "@jeremyparkphd")]
#
# Ships blank. Pass your own per run instead (--li, --x, --ig), or fill it in.
CREDIT = []
CREDIT_SIZE = 20                      # px at a 720px-wide panel; scales with the output
CREDIT_LINE_GAP = 6
CREDIT_MARGIN = 25                    # inset from the right and bottom edges
CREDIT_COLOR = (180, 180, 180)        # BGR
CREDIT_LABEL_COLOR = (120, 120, 120)  # the "LI:" / "X:" part
OUTPUT_CRF = 20

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached replies, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
