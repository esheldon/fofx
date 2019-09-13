# fofx
Astronomical friends of friends group finding using Source Extractor segmentation maps

## Example

```python
import fofx
import fitsio

seg = fitsio.read(seg_map_file)
fofs = fofx.get_fofs(seg)
```

## Requirements
- numpy
- numba
