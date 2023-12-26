from pathlib import Path

class Config:
	output_dir = Path("output")
	output_dir.mkdir(exist_ok=True)

	# preprocessing
	max_workers = 20
