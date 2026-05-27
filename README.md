# PyRod

I have a fair number of
[HO model train “cab ride” videos](https://www.youtube.com/playlist?list=PLjmlvzL_NxLr6EP-oa-RcA6YhSTHtBBFr)
on my [YouTube channel](http://youtube.com/c/raphaelmoll).
I particularly enjoy a view where the camera follows the last car, or precedes the front engine. Over the years I’ve been experimenting with various ways to achieve them, and I’m currently using this setup where a camera car is attached to the HO model train using a 3d-printed long rod:


![](https://www.alfray.com/ralf/blog/dev/img_be2145913ef488ba4ea58d6c17edd40ca437fc2343f2fbecc629edc95f58e937i.jpg 
"The camera car setup.")

For further details, you can read this
[blog post that explains the camera car setup](https://www.alfray.com/trains/blog/train/2023-01-02_cab_ride_videos_with_the_mob_a8befb95.html).

Back in 2023, I wrote a
[DaVinci Resolve plugin that “erases” the rod](https://www.alfray.com/trains/blog/train/2023-06-04_davinci_resolve_plugin_for_t_4126bb12.html),
which is very visible on camera and quite distracting:

![](https://www.alfray.com/ralf/blog/dev/img_774f73ea06bd597b20a57a8ae99d7c8ab7cb51d501b748af7910e1d2d33da8a9i.jpg 
"Original image captured by camera vs. Desired image for the final video")

**PyRod** is a re-implementation of this tool. However, instead of being a Lua plugin for DaVinci Resolve, it is now a standalone preprocessor Python command-line tool taking advantage of 
[numpy](https://numpy.org/) and
[OpenCV](https://opencv.org/)
for a faster and more complex image processing.

The command-line tool is designed to process the 4K video from a Mobius 4K camera and generate an output MP4 video suitable for DaVinci Resolve or any other non-linear editor.

Note that no AI or ML is involved here, this is just straight old fashioned image pixel processing.


## PyRod 1

[PyRod 1](./pyrod1/) runs a first pass that tries to analyze the image to find the bottom of the rod automatically.
It does that by computing a 
[Coefficient of Variation](https://en.wikipedia.org/wiki/Coefficient_of_variation)
on the bottom of the image to detect the smooth rod vs the rough ballast and ties:

![](https://www.alfray.com/ralf/blog/dev/img_e360ac4b6529de475a88c710e8334a11cb902963a7e2fa5b061a76d7488ba777i.jpg)

On the paper that was a good idea. In practice, it didn't work very reliably.

The second pass uses luminance thresholds and a flood fill algorithm to find the rod, trying to take advantage of its temporal stability between consecutive frames. Once again that doesn't quite work as easily as expected:

![](https://www.alfray.com/ralf/blog/dev/img_3cf10cdc8a15f483353155d04ff4888587665aa6ad928d65f4f32e5cd3f8a1bai.jpg 
"Red: the opaque rod mask.
Yellow: a smooth gaussian blur on each side for blending.
Green dot: the bottom center of the rod.")

The threshold-based masks are noisy and non-continuous. To get the desired mask, we apply a variety of OpenCV “[morphological transformations](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)”.

So overall, [PyRod 1](./pyrod1/) uses a bunch of experimental approached that, although interesting, didn't really pan out as desired. [PyRod 2](./pyrod2/) provides a better implementation.


## PyRod 2

[PyRod 2](./pyrod2/) uses pixel-based techniques which are very close to the original
[DaVinci Resolve plugin](https://www.alfray.com/trains/blog/train/2023-06-04_davinci_resolve_plugin_for_t_4126bb12.html) implementation.

The first pass of [PyRod 2](./pyrod2/) requires the user to select the coupler at the top of the rod manually. Once that pattern is memoryized and saved as JSON data, the tracker is implemented that locates its on every frame of the video:

![](https://www.alfray.com/ralf/blog/dev/img_ea0fd293407b7981b8d386544606b507e06fd8b4f00452a31efd2a2c16c040c5i.jpg
"The “heatmap” result of the matchTemplate() convolution.
Yellow: The region-of-interest search area.
Green: The matched coupler location.
0.91 is the convolution value at the peak (red spot).
0.28 is the quality score.")

Once we have the result of the pattern match convolution, a second pass analyzes the image under the coupler to locate the rod using a simple row-by-row luminance contrast approach:

![](https://www.alfray.com/ralf/blog/dev/img_d50d1c0f812d8caa73e8fb0733d905aa8de6c85b45031465ad46c7a7b7f637e0d.jpg)

The result of this analysis is used to create a polynomial approximation of the rod boundaries:

![](https://www.alfray.com/ralf/blog/dev/img_646a6dbcc712f600af8db28b374a73bfcdede7f34b1fd59d4971cc66576b6ba1i.jpg
"Dark blue rectangle is the coupler location.
Yellow curve: opaque center matching the rod.
Red curve: horizontal dilate to cover borders.
Green curve: A smooth blend on each side.")

and finally this is used to perform inpainting:

![](https://www.alfray.com/ralf/blog/dev/img_c7f2bda7a5442c6bb6207c9c9bfe83365939bdba48c40b2126c365d1a24bd65bi.jpg
"Inpaint left vs. Inpaint Mix vs. Inpaint Right")

~~
