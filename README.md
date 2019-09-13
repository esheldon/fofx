# fofx
Astronomical friends of friends group finding using Source Extractor segmentation maps

## Example

```python
import fofx
import fitsio

seg = fitsio.read(seg_map_file)
fofs = fofx.get_fofs(seg)
```

If the seg map looked like this then three objects would be grouped and one
would not

![Sample Png](https://raw.githubusercontent.com/esheldon/fofx/master/data/seg.png)


## Requirements
- numpy
- numba
