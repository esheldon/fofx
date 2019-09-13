# fofx
Astronomical friends of friends group finding using Source Extractor segmentation maps

## Example

```python
import fofx
import fitsio

seg = fitsio.read(seg_map_file)
fofs = fofx.get_fofs(seg)

for i in range(fofs.size):
    print(fofs['fof_id'][i], fofs['number'][i])

# for the seg map presented below this outputs
0      1
0      2
0      3
1      4
```

If the seg map looked like this then three objects would be grouped and one
would not

![Sample Png](https://raw.githubusercontent.com/esheldon/fofx/master/data/seg.png)

In particular if the ids were as follows
```
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 1 1 0 0 0 0 0 0 0 0 0
    0 1 1 1 0 0 0 0 0 3 0 0 0
    0 1 1 2 2 2 0 0 3 3 3 0 0
    0 0 2 2 2 2 2 3 3 3 3 3 0
    0 2 2 2 2 0 0 0 0 3 3 0 0
    0 0 2 2 2 0 0 0 0 0 0 0 0
    0 0 0 2 0 0 0 0 4 4 0 0 0
    0 0 0 0 0 0 0 0 4 4 4 0 0
    0 0 0 0 0 0 0 0 4 4 4 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
```
then the output of the get_fofs command would have the following structure
```
fof_id number
0      1
0      2
0      3
1      4
```


## Requirements
- numpy
- numba
