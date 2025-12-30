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