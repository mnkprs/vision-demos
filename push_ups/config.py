"""Params for the push-ups pose run.

Edit values here, then run `python main.py path/to/clip.mp4`, or point
INPUT_VIDEO at a clip and run `python main.py`. A few per-run flags override
them (`python main.py --help`): the clip, the output height, the trim and the
credit.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# The clip to run when none is passed on the command line. Film from the side,
# with the whole body in shot: hands, shoulders, hips and feet.
INPUT_VIDEO = DATA_DIR / "input" / "push_ups.mov"

# ── Conversion (.mov -> .mp4) ────────────────────────────────────────────────
# Always converted: ffmpeg applies the source rotation, OpenCV ignores it.
INFERENCE_HEIGHT = 1080   # uploaded to the model; ViTPose caps at a 2048px long edge
EXPORT_HEIGHT = 1080      # rendered; None keeps the source resolution
TRIM_SECONDS = None       # e.g. 3.0 for a fast test run
CONVERT_CRF = 23
FORCE_RECONVERT = False   # True re-runs ffmpeg even when a cached MP4 matches
REUSE_POSES = False       # True reuses cached poses instead of calling the gateway

# ── Model ────────────────────────────────────────────────────────────────────
MODEL = "usyd-community/vitpose-plus-large"
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0   # seconds; video pose is minutes, not seconds

EVERY_FRAME = True       # detect on every frame; costs wall clock, not money
VIDEO_FPS = 2.0          # detector cadence, used only when EVERY_FRAME is False
VIDEO_MAX_FRAMES = None  # None -> the model's 900-frame default
PRECISION = 4            # decimal places on normalized coordinates (1-8)

# ── Render ───────────────────────────────────────────────────────────────────
DRAW_FACE = False             # face keypoints read as a scribble over the face
MAX_PERSONS = 1               # one body should be one subject
PERSON_IOU_THRESHOLD = 0.6    # above this, two boxes are the same person
DRAW_BBOX = True
BBOX_COLOR = (255, 160, 40)   # BGR, blue
DRAW_TRACK_LABEL = False      # the "id 0" tag above the box
DRAW_HUD = False              # the frame/person counter in the top-left
LINE_THICKNESS = 3            # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4
OUTPUT_CRF = 20

# Skeleton color per limb, BGR, keyed like pull_ups':
#   {"left_arm": ..., "right_arm": ..., "left_leg": ..., "right_leg": ..., "torso": ...}
# None keeps the per-side colors (left cyan, right orange, torso green) and rings
# each joint in its track's color.
SKELETON_COLORS = None

# ── Plank gate ───────────────────────────────────────────────────────────────
# Only frames in a plank say anything about push-ups: standing before the set or
# walking off after it would otherwise read as a huge rep. The torso (mid-shoulder
# to mid-hip) has to lie within this many degrees of horizontal.
PLANK_MAX_TORSO_DEGREES = 45.0
PLANK_MIN_SECONDS = 0.5   # shorter plank runs are a body passing through, not a set

# ── Rep counting ─────────────────────────────────────────────────────────────
# The signal is `wrist_y - shoulder_y`, read as depth below the top plateau; see
# rep-counting-explained.md for the method.
COUNT_REPS = True

SHOULDER_JOINTS = ("left_shoulder", "right_shoulder")
WRIST_JOINTS = ("left_wrist", "right_wrist")
HIP_JOINTS = ("left_hip", "right_hip")   # for the torso angle the plank gate reads
# Joints under this confidence are left out of the means. From the side the far
# limbs are guesses, so the near side carries the signal.
MIN_KEYPOINT_SCORE = 0.3

SMOOTH_MEDIAN_FRAMES = 5     # median filter, to kill isolated spikes
SMOOTH_MEAN_FRAMES = 7       # then a centered moving average, to kill jitter
TOP_PERCENTILE = 90          # relative zero: the top plateau, not the single highest frame
REP_DEPTH_FRACTION = 0.45    # sink past this share of a typical rep's depth to confirm a rep
REP_RESET_FRACTION = 0.20    # come back within this share of the top to close it

# Where the down phase starts and the up phase ends, so a pause at the top is
# not counted as either.
REP_PHASE_METHOD = "velocity"        # or "threshold"; see rep-counting-explained.md
REP_PHASE_VELOCITY_FRACTION = 0.10   # lower = phases reach further into the top plateau
REP_PHASE_FRACTION = 0.05            # used only by REP_PHASE_METHOD = "threshold"

# ── Depth and body line ──────────────────────────────────────────────────────
# A rep is full when its smallest elbow angle (shoulder-elbow-wrist) reaches this.
FULL_DEPTH_ELBOW_DEG = 90.0

# Shoulder-elbow-wrist per side. Each rep reads the side the model was surest of.
ARM_JOINTS = (("left_shoulder", "left_elbow", "left_wrist"),
              ("right_shoulder", "right_elbow", "right_wrist"))

# Shoulder, hip, knee, ankle per side. The body line is the hip angle at the
# bottom, shoulder-hip-ankle, or shoulder-hip-knee when the ankle is unclear.
BODY_LINE_JOINTS = (("left_shoulder", "left_hip", "left_knee", "left_ankle"),
                    ("right_shoulder", "right_hip", "right_knee", "right_ankle"))
# Read on every frame of the rep and the rest at the top after it; the verdict is
# the furthest the hips stayed off the line for BODY_LINE_HOLD_SECONDS running.
BODY_LINE_TOLERANCE_DEG = 10.0   # off straight by more than this: sagging, or hips high
BODY_LINE_HOLD_SECONDS = 0.12    # shorter excursions are pose noise, not a body position
BODY_LINE_ZONE_FRACTION = 0.15   # this close to either end of the rep reads "top" or "bottom"
ANKLE_MIN_SCORE = 0.3            # mean ankle confidence below this: read to the knee

GRADE_SAMPLE_FRAMES = 5   # median window, so one bad pose can't decide a grade

# ── The reported number ──────────────────────────────────────────────────────
# What the graph plots per rep, and what the closing note ranks.
#   "depth": the elbow angle at the bottom, in degrees; lower is deeper
#   "total": down + up, in seconds
#   "down":  top to bottom, the eccentric
#   "up":    bottom to top, the concentric
REP_METRIC = "depth"

# ── Side panel ───────────────────────────────────────────────────────────────
# The clip on the left, the stats panel on the right: the same width as a
# portrait clip, capped at 9:16 for a landscape one so the text still fits.
# Top to bottom: the headline numbers, the live block, the graph, the closing note.
SIDE_PANEL = True
PANEL_PROGRESSIVE = True   # False draws every rep from frame 0 instead
PANEL_PLOT_SCALE = 0.76    # square graph edge, as a fraction of the panel's short side
PANEL_Y_TICKS = 4          # target y-axis tick count on the graph

# The live block, redrawn every frame: elbow angle with a dial, the phase of the
# rep, the rep timer, the body line, and depth (0% arms locked at the top, 100%
# shoulders at the height of the hands).
PANEL_LIVE = True
PANEL_LIVE_SMOOTH_FRAMES = 5   # trailing median on the live values; never reads ahead

# Band heights, px at a 720px-wide panel. Spare height on a tall panel is spread
# evenly between them; a short one takes it out of the graph.
PANEL_HEADER_BAND = 230
PANEL_LIVE_BAND = 300     # 0 when PANEL_LIVE is False
PANEL_NOTE_BAND = 130

# Text sizes and gaps are px at a 720px-wide panel and scale with the output.
# A header or note band that cannot hold its lines at this size shrinks them to fit.
PANEL_HEADER_SIZE = 34
PANEL_NOTE_SIZE = 28           # the closing note
PANEL_HEADER_LINE_GAP = 0.45   # line spacing, as a multiple of the font size
PANEL_NOTE_LINE_GAP = 0.65
PANEL_LIVE_NUMBER_SIZE = 80    # the live elbow angle
PANEL_LIVE_SIZE = 30           # phase and rep timer
PANEL_LIVE_CAPTION_SIZE = 20   # the captions, the depth percentage
PANEL_LIVE_LINE_SIZE = 24      # the live body-line readout

REP_LABEL = "Push-ups"
PANEL_AVG_LABEL = "Avg rep"
PANEL_LAST_LABEL = "Last rep"
PANEL_FULL_LABEL = "Full"
PANEL_LIVE_ELBOW_LABEL = "Elbow"              # followed by ", left arm"
PANEL_LIVE_TIMER_LABEL = "Rep"                # "Rep 3: 1.24s"
PANEL_LIVE_DEPTH_LABEL = "Depth"
PANEL_LIVE_LINE_LABEL = "Body line"
PANEL_GRAPH_TITLE = None   # None follows REP_METRIC, e.g. "Elbow at Bottom (°)"
PANEL_Y_LABEL = None       # likewise

PANEL_FONT = "auto"        # "auto", "opencv" to force Hershey, or a font path
PANEL_FONT_INDEX = None    # face index inside a .ttc; None uses the default

# Credit lines, bottom-right of the export: ("LABEL", "value") or a plain string.
#   CREDIT = ["Jeremy Park"]
#   CREDIT = [("LI", "Jeremy Park, PhD"), ("X", "@jeremyparkphd")]
# Ships blank; fill it in or pass --li, --x, --ig per run.
CREDIT = []
CREDIT_SIZE = 20               # px at a 720px-wide reference; scales with the output
CREDIT_LINE_GAP = 6            # px between credit lines
CREDIT_MARGIN = 25             # inset from the right and bottom edges
CREDIT_OPACITY = 0.75          # 0-1; white, dialled back so it reads as a credit
CREDIT_LABEL_OPACITY = 0.41    # the "LI:" / "X:" part, a shade dimmer

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"     # converted MP4s + cached poses, reused across runs
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"

SAVE_JOINT_PLOT = False      # one joint's height over time, one line per track
PLOT_JOINT = "left_shoulder" # any name from src.skeleton.KPT_NAMES
