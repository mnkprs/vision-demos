"""Params for the pull-ups run: pose, reps, and each rep's range of motion.

Edit values here, then run `python main.py`.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# A clip, or a folder whose top-level videos run in turn (subfolders skipped).
# Clips or folders on the command line replace it; `python main.py -h` lists flags.
INPUT = DATA_DIR / "input"
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted, since OpenCV ignores source rotation. Poses are normalized,
# so changing EXPORT_HEIGHT needs no new gateway call (ViTPose caps at 2048px).
INFERENCE_HEIGHT = 1080   # what the model sees; changing it re-runs pose
EXPORT_HEIGHT = 1080      # the rendered video; None keeps the source resolution
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23           # x264 quality for the cached conversions; lower is better
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches

# ── HDR ──────────────────────────────────────────────────────────────────────
# iPhone HDR (HLG) looks washed out in OpenCV. "auto" tone-maps HDR-tagged
# sources, "off" never, "force" always. Exported video only.
TONEMAP = "auto"

# `bt.2446a` is the safe default; `spline` keeps midtones brighter, `hable` is filmic.
TONEMAP_ALGORITHM = "bt.2446a"

# True also feeds the model tone-mapped frames (re-runs pose).
TONEMAP_INFERENCE = False

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

# True reuses a cached pose reply when its inputs match; any changed setting misses.
REUSE_POSES = True

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
EVERY_FRAME = True       # detect on every frame; costs wall clock, not money
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)
MAX_PERSONS = 1          # one athlete
PERSON_IOU_THRESHOLD = 0.6  # box overlap above which two detections are one person

DRAW_FACE = False        # face keypoints read as a scribble over the face
# Draw the person's bounding box.
DRAW_BBOX = False
BBOX_COLOR = (255, 160, 40)   # BGR, blue
LINE_THICKNESS = 3       # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4         # keypoint dot radius, px at a 720px-wide frame

# Skeleton color per limb, BGR; the torso stays neutral so the limbs carry color.
SKELETON_COLORS = {
    "left_arm":  (238, 211, 34),    # cyan     #22D3EE
    "right_arm": (241, 102, 99),    # indigo   #6366F1
    "left_leg":  (249, 121, 232),   # magenta  #E879F9
    "right_leg": (133, 113, 251),   # rose     #FB7185
    "torso":     (225, 213, 203),   # slate    #CBD5E1
}

# An arc at each elbow showing the upper arm-forearm angle.
DRAW_ELBOW_ANGLES = True
ELBOW_ARC_COLOR = (230, 230, 230)   # BGR
ELBOW_ARC_RADIUS = 24    # px at a 720px-wide frame; scales with the export
# True colors the arc by each rep's bottom grade (PANEL_FULL or PANEL_PARTIAL)
# while the athlete is at that bottom, where the hang is measured.
ELBOW_ARC_GRADE_COLORS = False

# ── Reps ─────────────────────────────────────────────────────────────────────
# The signal is `hand_y - torso_y`; see how-it-works.md for the method.
TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
HAND_JOINTS = ("left_wrist", "right_wrist")
# Head point: confidence-weighted mean of these, skipping scores under
# HEAD_MIN_SCORE. From behind the ears dominate; from the front, eyes and nose.
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
HEAD_MIN_SCORE = 0.3

REFERENCE_TO_HANDS = True         # False measures against the frame instead
HAND_OFF_BAR_TOLERANCE = 0.05     # wrist drift allowed before "hands off the bar"
SMOOTH_MEDIAN_FRAMES = 5          # median filter, to kill isolated spikes
SMOOTH_MEAN_FRAMES = 7            # then a centered moving average, to kill jitter
BASELINE_PERCENTILE = 10          # relative zero: the hanging plateau, not the minimum
REP_PEAK_FRACTION = 0.60          # rise past this share of rep height to confirm a rep
REP_RESET_FRACTION = 0.30         # fall back under this to arm the next one

# Looser height gate when calibrated, so half reps still get counted and graded;
# the flexion check below rejects wobble instead.
REP_PEAK_FRACTION_CALIBRATED = 0.35

# Elbow flexion a calibrated rep needs, as a share of the reference sweep.
# Raise it to ignore small attempts, lower it to count every twitch.
REP_MIN_FLEXION_FRACTION = 0.35

# False counts every pull; range of motion is graded separately, below.
REQUIRE_HEAD_ABOVE_HANDS = False

# Which span the PULL column, the average and timeline.png all report.
#   "moving": onset to peak, excluding any dead hang at the bottom
#   "ascent": valley to peak, hang included
#   "total":  valley to valley, the whole cycle
REP_METRIC = "moving"
REP_ONSET_METHOD = "velocity"        # or "threshold"; see how-it-works.md
REP_ONSET_VELOCITY_FRACTION = 0.10   # lower = starts earlier, nearer the base
REP_ONSET_FRACTION = 0.05            # used only by REP_ONSET_METHOD = "threshold"

# ── Range of motion ──────────────────────────────────────────────────────────
# Each rep is graded full or partial at its own valley and peak; see how-it-works.md.
GRADE_ROM = True

# What a full rep looks like for this athlete, built once with:
#   python calibrate.py data/input/my_full_set.mov
# Settings marked "uncalibrated" below are the fallback when it is absent.
CALIBRATION_PROFILE = DATA_DIR / "calibration.json"
USE_CALIBRATION = True    # False grades every clip against itself (--uncalibrated)
CALIBRATION_VIDEO = DATA_DIR / "input" / "pull_ups_full_rom.mov"   # calibrate.py's default

# Reference-clip sanity checks, warned about rather than enforced: max hang
# spread across reps (degrees), and minimum rep count.
CALIBRATION_MAX_SPREAD_DEGREES = 6.0
CALIBRATION_MIN_REPS = 3

# Shoulder-elbow-wrist, left arm first. The grade reads their mean.
ARM_JOINTS = (("left_shoulder", "left_elbow", "left_wrist"),
              ("right_shoulder", "right_elbow", "right_wrist"))

# Bottom: degrees the valley elbow angle may fall below the reference hang (the
# profile's, else this clip's deepest). A real dead hang reads ~167, not 180.
ROM_HANG_TOLERANCE_DEGREES = 12.5

# Top: head clearance over the hands needed, as a share of the reference clip's.
ROM_CLEARANCE_FRACTION = 0.40

# ── ...and the same two, uncalibrated ────────────────────────────────────────
# Used only when no profile has been built.
#
# Minimum hang elbow angle, in degrees, so a clip of only short reps can't pass.
ROM_HANG_ELBOW_DEGREES = 150.0

# Head above hands, in torso lengths. Roughly 0.10 puts the chin at the bar.
ROM_CLEARANCE_TORSO_FRACTION = 0.18

# Frames in the median window around each extreme, so one bad pose can't decide a rep.
ROM_SAMPLE_FRAMES = 5

# ── Side panel ───────────────────────────────────────────────────────────────
# The clip on the left, the stats panel on the right: the same width as a
# portrait clip, capped at 9:16 for a landscape one so the text still fits.
# False writes the overlaid clip alone, at its own size (skeleton and elbow
# arcs, nothing beside it); timeline.png and the JSON files are unchanged.
SIDE_PANEL = True
PANEL_HEADER_SIZE = 44     # px at a 720px-wide panel; scales with the output
PANEL_LIVE_SIZE = 34       # the "Full ROM" line and the live elbow angles
PANEL_ROW_SIZE = 26        # the per-rep rows
PANEL_NOTE_SIZE = 20       # column headings
# Display-only Gaussian smoothing (sigma, seconds) of the graph and live elbow
# value; grades use the raw series. 0 disables it.
PANEL_ANGLE_SMOOTH_SECONDS = 0.1
PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

# Panel colors, BGR. FULL and PARTIAL mark each rep's grade; ACCENT is the
# displacement line and its dot, PLAYHEAD the vertical line at the current frame.
PANEL_BG = (26, 22, 20)          # #14161a
PANEL_FG = (247, 244, 242)       # #f2f4f7
PANEL_MUTED = (153, 145, 139)    # #8b9199
PANEL_GRID = (56, 48, 43)        # #2b3038
PANEL_FULL = (102, 207, 81)      # #51cf66
PANEL_PARTIAL = (107, 107, 255)  # #ff6b6b
PANEL_ACCENT = (255, 160, 40)    # #28a0ff
PANEL_PLAYHEAD = (153, 145, 139) # #8b9199

# Credit lines, bottom-right of the panel: ("LABEL", "value") or a plain string.
#   CREDIT = ["Jeremy Park"]
#   CREDIT = [("LI", "Jeremy Park, PhD"), ("X", "@jeremyparkphd")]
# Ships blank; fill it in or pass --li, --x, --ig per run.
CREDIT = []
CREDIT_SIZE = 20                      # px at a 720px-wide panel; scales with the output
CREDIT_LINE_GAP = 6                   # px between credit lines
CREDIT_MARGIN = 25                    # inset from the right and bottom edges
CREDIT_COLOR = (180, 180, 180)        # BGR
CREDIT_LABEL_COLOR = (120, 120, 120)  # the "LI:" / "X:" part
OUTPUT_CRF = 20                       # x264 quality of the rendered video

# ── Timeline (timeline.png) ──────────────────────────────────────────────────
# Line and mark colors by role, BGR. "full" and "partial" mark each rep's grade.
TIMELINE_COLORS = {
    "full":      (68, 158, 47),    # #2f9e44
    "partial":   (49, 49, 224),    # #e03131
    "line":      (246, 109, 47),   # #2f6df6, smoothed displacement and the rep spans
    "raw":       (214, 206, 201),  # #c9ced6
    "interval":  (171, 100, 24),   # #1864ab, each rep's bracket and its seconds
    "peak":      (12, 89, 232),    # #e8590c, each rep's top and its time
    "elbow":     (196, 61, 95),    # #5f3dc4, the graded mean elbow angle
    "clearance": (62, 138, 43),    # #2b8a3e, head above hands
    "cutoff":    (42, 42, 201),    # #c92a2a, what a full rep must reach
    "label":     (150, 142, 134),  # #868e96, dead-hang labels and the uncalibrated floor
    "onset":     (112, 100, 92),   # #5c6470, each rep's onset time
}
# None keeps matplotlib's white figure. A dict of "background", "text" and
# "grid" colors (BGR) restyles it, e.g. dark to sit beside the video.
TIMELINE_FIGURE = None

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
