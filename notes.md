# Notes

# Version 1:

Static does not work well when purely random. Original structure is still clearly visible, although not sure how this would affest an AI.

Static does convert back to original image properly, so transparent \<img\> elements would work.

Perhaps reigonal static, since having static in an area
allows the human eye to average the image out automatically.

As another note, this static division method will
eventually drive some of the images to their
maximum or minimum values, from where they can be changed no more. Thie may account for some of the problems-- maybe there is an advanced image-to-static library out there?

# Version 2:

Starting with pure random noise doesn't seem to work completely.
If we could have arbitrarily large / small pixel values in images, it
would be a lot easier.

I feel like if the noise was in large chunks, or perhaps LODs,
it would work better. Perhaps lots of polygons of color,
when overlapping in just the right spots, will recreate the images
properly.

Perhaps randomly shaped chunks of the image could be removed to different layers, no noise applied, could work. Both only reveal the final image when the whole thing is put together. The only downside is that some of the style may seep through, depending on the size of the chunks taken

# Version 2.5 + 3:

Can't get 2.5 to make brighter noise, it seems...

And as for 3, yeah, the noise is a bit better, but still
not good enough.

The best lead I have so far is to have different layers of pure noise, and not rely on the pixels averaging well,
but just up and change the transparency of each so it has to average well. I dunno if an AI will bother to read the alpha channels to clearly, and maybe avoiding individual-channel averages will help get a cleaner (or, well, messier) result.

# Version 5:

First working version! Got normal and multiply blend-modes working (multiply looks promising!) and a LOD-based recreation system that works pretty well. Plus html demos to test out the system!

Hope to add more blend mode support in the future.

Would love css animations for the demos too...

# Version 5.1

Got screen mode blend working, always forget to change
pixellation mode in the new methods... This one is better than multiply. Very excited!

# Version 5.2.1

We now have normal, lighten, darken, plus-lighter, multiply, and screen blend modes, and the code is much simpler. Getting this to work in JS will prove an issue, as there doesn't seem to be a good equivalent of numpy for JS.

Although [Tensorflow.js](https://github.com/tensorflow/tfjs) seems like a good option, with full array capabilities
[StdLib](https://github.com/stdlib-js/stdlib) seems very promising too.

# Version 6:

JS Demo is functional, and semi-efficient, although could stand some asynchronous processing.