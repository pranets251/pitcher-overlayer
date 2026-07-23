# Pitcher Overlayer

Local, video-first coaching tool for front-facing bullpen footage. It detects individual delivery windows and exports one synchronized overlay for each consecutive pair in chronological order. Pitcher pose matching inside a constrained release window corrects frame offsets, and background registration reduces small camera shifts before clips are blended.

## Run

```bash
cd "pitcher overlayer"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analyze.py data/PitcherVideo.mov --model vendor/chonyy-overlay/model/yolov4-tiny-baseball-416 --out output/analysis.json
.venv/bin/python app.py
```

Open <http://127.0.0.1:5177>. The page loads the bundled bullpen result and accepts MOV, MP4, M4V, or AVI uploads for a new local analysis.

## Public deployment

The included Dockerfile runs the app behind Gunicorn. Deploy it only on a host with persistent disk: uploaded jobs, generated videos, and content-hash calibration files are stored under `jobs/` and `calibrations/`. Each upload receives an isolated job ID, so users do not overwrite one another. The bundled `demo/` session remains the default for new visitors.

This workload is CPU/GPU- and storage-intensive. Add authentication, request-rate limiting, quotas, and scheduled job cleanup at the hosting layer before sharing a broadly public URL.

## Interpretation

The tool does not detect or track the baseball, infer pitch types, cluster paths, calculate tunneling scores, rank pairs, add tracers, or color the baseball.

## Deploy on Render

This folder includes a Dockerfile and `render.yaml`. Deploy the folder as its own
Git repository, then create a Render Blueprint from that repository. The Blueprint
uses Render's free web-service tier. Render supplies the public HTTPS
`onrender.com` address after the first deploy. Free instances use ephemeral storage,
so uploaded and processed sessions can disappear after a restart or redeploy; the
bundled default demo remains part of the application image.
