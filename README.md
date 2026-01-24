# ⚔️ Katana

<p align="center">
  <img src="assets/Letter_Logo.svg" alt="Katana Letter Logo" width="100">
  <img src="assets/Robot_Logo.svg" alt="Katana Robot Logo" width="100">
</p>

## 📌 What it is
Katana is a system designed to confuse image scraping bots, such as those used by OpenAI and Google. 

Any art placed on the Web is in danger of being scraped and used to train AI models. Tools
such as Glaze and Nightshade exist, but they modify the image, however imperceptably, and can be undone just by blurring the image slightly. Katana keeps the image exactly as it is, but to any web scraper, all it sees are pixellated, distorted, and heavily discolored images. 


- 💻 Originally made for [purelyhuman.xyz](https://purelyhuman.xyz/)

- [📝 Read the article](https://medium.com/@joshward_accounts/how-an-obscure-css-feature-can-protect-your-art-from-ai-f5169ea5ff9a)

- [🎧 Check out the Discord thread!](https://discord.com/channels/1116110158775468054/1455285065037910077)

## 📌 Why choose Katana?

### - 🍃 Katana is lightweight

Less than 500 lines of code, and much less computationally expensive than Glaze or Nightshade

### - 💻 Katana is fast

For large images, with shuffling enabled, Katana takes only a second or two to generate layers

### - 🎯 Katana is effective

Katana has the freedom to appply very destructive edits to the source image, a freedom which standard AI poison systems do not have

### - 🎨 Katana is versatile

Katana can operate on different blend modes to more effectively obfuscate different parts of an image

## 📌 How it works

Katana works by slicing images into layers of useless noise / discolored images, 
that when laid over each other with the correct blend mode, recreate the original image, 
although there is no single \<img\> element that contains the actual image.

In more detail, the system:

1. Pads the image to be a square with both sides a power of 2
2. Generates a series of special LODs of the image
3. Generates inbetweens: images that, when added to one LOD, result in the next one
4. Rescales and unpads these inbetweens
5. Optionally shuffles parts of the inbetweens between each other, for heavy obfuscation

None of the layers are really usable until they are layered on top of each other with the correct blend mode.

This causes a massive amount of difficulty for web scrapers that intend to use the images for AI training purposes (or any other), as the scraper must be explicitly programmed to combine images in the correct way, or to screenshot the page and somehow isolate the image part. If a scraper scrapes a massive amount of these images, and the images' metadata has no information telling what images go to each other, it will be nearly impossible to reconstruct the original images again.

Future versions of Katana may split the image into chunks and give each their own div, or even have heavily poisoned parts of layers that are covered up by later layers.