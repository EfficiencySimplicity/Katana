# DevNotes: Notes for developers

## Thank you for choosing Katana for your site!

This file contains helpful tips and best practices for getting the most out of Katana!

(NOTE: Katana is still in-dev. This version is a demo. It is recommended you do not use)

### The basic data transfer should follow this format:

- Server sends (encrypted) plain image
- Client attempts to run Katana on image
- If client cannot, client pings server to run Katana on its side
- Server runs most of Katana, and sends unclipped LOD inbetweens
- Client attempts to clip, and if it cannot, it pings the server again.
- The server then clips the LOD inbetweens it has and sends the full images over
- Client populated the page with the clipped Katana layers

### The main steps of the Katana process go like this:

- Plain image is padded with 0s to be a square, with each side being a power of 2
- LODs of the padded image are generated, at varying sizes.
- Inbetweens are generated from the LODs.
- These inbetweens are scaled to be the same size as the padded image
- These inbetweens have their padding clipped off to match the original image

- Optionally (So very recommended) swap sections of the inbetweens with each other