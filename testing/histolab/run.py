
import os

from histolab.slide import Slide
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer

fp = "/opt/ml/input/data/train/Mild/84440T_001.tif"
out_dir = os.getenv('SM_MODEL_DIR')

for path in [fp, out_dir]:
    if not os.path.exists(path):
        raise Exception(path)

slide = Slide(
    fp,
    processed_path=out_dir
)

print(f"{slide.name = }")
print(f"{slide.levels = }")
print(f"{slide.dimensions = }")

thumbnail = slide.thumbnail
thumbnail_path = os.path.join(out_dir, "thumbnail.png")
thumbnail.save(thumbnail_path)

tiler = ScoreTiler(
    scorer=NucleiScorer(),
    tile_size=(512, 512),
    n_tiles=100,
    level=0,
    check_tissue=True,
    tissue_percent=80.0,
    prefix="tiles/",
    suffix=".png"
)

tiler.extract(slide)
