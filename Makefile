PY := uv run
POSITIONS := data/emoji_positions.json

.PHONY: all positions force-positions clean serve

all: positions

positions: $(POSITIONS)

$(POSITIONS): compute_positions.py
	$(PY) compute_positions.py --out $(POSITIONS)

force-positions:
	$(PY) compute_positions.py --out $(POSITIONS) --force

serve:
	$(PY) -m http.server 8000

clean:
	rm -f $(POSITIONS)
