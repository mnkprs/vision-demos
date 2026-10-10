# How a rep is found, timed and graded

1. Run pose estimation on every frame to get the body's keypoints.
2. Keep only the frames where the body is in a plank.
3. Measure how far the shoulders drop below the top, against the hands on the
   floor.
4. Count a rep each time the shoulders sink and come back up.
5. Time the way down and the way up, grade the depth from the elbow angle at
   the bottom, and follow the body line from the hip angle through the whole
   rep.

Code: [`src/reps.py`](src/reps.py). Every parameter, and why each default is
what it is: [`config.py`](config.py). Per-rep numbers land in `reps.json` and
`summary.txt`.

## The plank gate

- The torso is the line from mid-shoulder to mid-hip. A frame counts only when
  that line is within `PLANK_MAX_TORSO_DEGREES` (45°) of horizontal.
- Standing, kneeling up and walking all leave the torso near vertical, so
  getting into position and walking off afterwards never read as a rep.
- Plank stretches shorter than `PLANK_MIN_SECONDS` are dropped too: that is a
  body passing through horizontal, not a set.
- `displacement.png` greys out the ignored frames.

## The signal

- `wrist_y - shoulder_y`: the shoulders' height above the wrists. The hands
  are planted on the floor, so they are a fixed point the way the hands on the
  bar are for chin-ups, and a handheld camera does not register as movement.
- Shoulders and wrists are each a confidence-weighted mean of both sides. From
  the side, the far arm is mostly the model's guess, so the near arm, which it
  is surer of, carries the signal.
- A median filter then a centered moving average: spikes, then jitter, without
  shifting the turns.
- A push-up rests at the **top**, so the signal is read the other way up from a
  chin-up. Zero is the top plateau (a high percentile, so one frame can't set
  it), and each frame's **depth** is how far the shoulders sit below it. A rep
  becomes a rise in depth and back, which the counter reads directly.

## The count

- A hysteresis state machine on depth: armed at the top, confirmed once the
  shoulders sink past `REP_DEPTH_FRACTION` of a typical rep's depth, closed when
  they come back within `REP_RESET_FRACTION` of the top. Two thresholds rather
  than one, or noise wobbling across a single line counts several times.
- The confirm line is set low (45%) on purpose: a shallow rep still counts, and
  is then graded partial, instead of silently vanishing.
- A rep needs a top before it, inside one unbroken plank. A clip that opens at
  the bottom does not count its first press, since its way down was never seen;
  a rep left at the bottom when the plank breaks or the clip ends was never
  pressed back up, so it does not count either.

## The timing

- **Down** (eccentric): from where the descent starts to the bottom.
- **Up** (concentric): from the bottom to where the press locks out.
- **Total**: down + up. A pause at the top is in neither.
- The start of the descent is found by walking back from its fastest point
  until the curve flattens; the lockout is the same rule run backwards in time,
  so both ends use one method.

## The grades

- **Depth**: the smallest elbow angle (shoulder–elbow–wrist) in the rep, read on
  the arm the model was surer of over that rep. At or under
  `FULL_DEPTH_ELBOW_DEG` (90°) it's **full**; otherwise **partial**. The angle
  is read to the whole degree, the precision the panel shows, so a rep shown at
  90° is never failed on 90.4°. An elbow the model couldn't see counts as
  partial, since assuming depth would flatter the athlete.
- **Body line**: the hip angle, shoulder–hip–ankle, as degrees off straight,
  read on every frame of the rep and of the rest at the top after it. Hips
  below the shoulder-to-ankle line are **sagging**, above it **hips high**
  (piked). Each rep reports the furthest the hips stayed off the line for at
  least `BODY_LINE_HOLD_SECONDS` running (so a one-frame spike can't decide
  it), which side of the line, and where in the rep: top, going down, bottom
  or pushing up. Beyond `BODY_LINE_TOLERANCE_DEG` it's flagged.
- **Why 10°**: on the sample clip, filmed from three-quarters in front, a body
  that looks straight at the bottom reads 2–7° off, and a top where the hips
  visibly hump above the line reads 11–13°. 10° falls between the two. Pose
  noise alone is a few degrees, so much lower would flag straight reps; much
  higher would miss a pike you can see.
- **Ankles not visible enough**: when the ankle's mean confidence over the rep
  is under `ANKLE_MIN_SCORE`, often because the feet are cropped or hidden, the
  line is read to the knee instead (shoulder–hip–knee). `summary.txt` says which
  joint each rep used. Hip to knee is a shorter lever, so it tells you about
  the hips but nothing about a bend at the knees.
- Both readings are smoothed with a median over a few frames
  (`GRADE_SAMPLE_FRAMES`), so one bad pose can't decide a grade.

## What the panel shows

- **Header:** reps so far, the full-depth tally (green while every rep reached
  depth), the average rep time and the last rep's.
- **Graph:** one dot per rep, colored by its grade. `REP_METRIC` picks what it
  plots: `depth`, the elbow angle at the bottom with the 90° cutoff drawn in
  (the default), or `total`, `down` or `up` in seconds.
- **Closing note:** the deepest and shallowest rep (or fastest and slowest),
  and a one-line body-line verdict, e.g. "Hips high, reps 1–5 (13° at top)".

### The live block

Between the header and the graph, redrawn every frame (`PANEL_LIVE`):

- **Elbow angle**, as a number and a dial whose white mark is the full-depth
  cutoff. It reads the arm the rep is graded on, so at the bottom it settles on
  the rep's own number. It is smoothed with a median over the last
  `PANEL_LIVE_SMOOTH_FRAMES` frames, which looks back only and lags by about
  two frames.
- **Phase**: *going down* from a rep's onset to its bottom, *pushing up* from
  the bottom to lockout, *resting at top* between reps. These are the same
  onset, bottom and lockout as the per-rep table, so the two never disagree. A
  dash before the first rep and outside a plank.
- **Rep timer**: runs from the rep's onset, stops at its total on lockout, and
  starts again at the next onset.
- **Body line**: *straight* in green, *hips high* or *hips sagging* in red with
  the degrees off the line. Same side, foot reference and cutoff as the rep's
  verdict, smoothed looking back, and a new reading shows only once it has held
  for `BODY_LINE_HOLD_SECONDS`, so it doesn't flicker at the cutoff.
- **Depth**: measured on the body, so it reads the same on every rep, the
  first included. 0% is the top plateau, arms locked; 100% is the shoulders
  down at the height of the hands. The plateau is the same high percentile the
  rep signal uses, taken over the plank frames up to the current one, so the
  bar never reads a frame that hasn't happened yet. A chest-to-floor rep stops
  short of 100%, since the shoulder joint stays a few inches above the hands.

The per-rep layer is still drawn once per rep count with matplotlib; the live
block is OpenCV and Pillow over a background drawn once, so it costs about a
millisecond a frame.

## Limits

- **The back itself is invisible.** The pose model has no keypoints along the
  spine: only shoulder, hip, knee and ankle. The body-line check measures where
  the hips sit against the shoulder-to-ankle line, nothing more. A rounded or
  arched lower back with the hips still on that line reads as straight. The
  deadlift project reads back shape by asking a vision model about each frame
  ([`../deadlift/src/classify.py`](../deadlift/src/classify.py)); the same
  approach is the next step here.
- Film from the side. Head-on, the elbow angle and the hip angle are
  foreshortened, and both read straighter than they are.
- From three-quarters in front, the count and the times hold up but the elbow
  angle does not: at the bottom ViTPose can put the shoulder on the front of the
  chest and the elbow on the deltoid, which reads about 90° whatever the real
  depth. The overlay shows it, so check the skeleton at the bottom of a rep
  before trusting a depth grade.
- Angles are measured in the image, so a camera at an angle to the athlete
  bends them a little; ViTPose's shoulder sits on the deltoid, not in the joint.
  Check a rep near 90° by eye before trusting the grade either way.
- One subject: the signal reads the tracked person, so a second body in frame is
  ignored.
- Depth is calibrated per clip, so it does not compare across videos. The elbow
  angle does.
