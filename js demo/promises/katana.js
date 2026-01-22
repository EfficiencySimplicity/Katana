// JS DEMO



/**
 * BASICS
 */


// TODO: padding near each other

/**
 * Returns the side length of the smallest square that can contain the image, and that has a side length that is a power of 2
 * @param   {tf.Tensor} image An image tensor
 * @returns {Number}          The padded size
 */
function getPaddedSize(image) {
    let [w, h, c] = image.shape
    return Math.ceil(2**Math.ceil(Math.log2(Math.max(w,h))))
}


/**
 * Pixellates an image tensor with the correct depth and blend mode
 * @param   {tf.Tensor} image     A square, 3-channel, 0-1 tensor representing an image
 * @param   {Number}    depth     The depth to pixellate to. The size of the result is 2^depth
 * @param   {String}    blendmode The blend mode used for pixellation
 * @returns {tf.Tensor}           The scaled-down tensor
 */
function pixellate(image, depth=1, blendmode='screen') {return tf.tidy(() => {

    let [w, h, c] = image.shape
    let d         = Math.floor(2 ** depth)

    // Reshape it into (d, d, chunk_size, chunk_size, 3)
    let pixel_map = image.reshape([d, Math.floor(w/d), d, Math.floor(h/d), c])
    pixel_map     = tf.einsum('abcde->acbde', pixel_map);

    // Each function (min/max) collapses each chunk into a single value,
    // Resulting in shape (d,d,3)
    if (['multiply', 'darken'].includes(blendmode)){
        return pixel_map.max(axis = [2,3])
    }else if (['screen', 'lighten', 'plus-lighter'].includes(blendmode)){
        return pixel_map.min(axis = [2,3])
    }

})}


/**
 * Upscales an image tensor
 * @param {tf.Tensor} image  The image tensor to upscale
 * @param {Number}    factor The upscaling factor
 * @returns                  The image, upscaled by the factor
 */
function scaleBy(image, factor=2) {return tf.tidy(() => {
    // Just create a whole layer object! What could go wrong?
    // Needless to day, clean this. Won't it always be 2?
    let [w, h, c] = image.shape
    let upsampler = tf.layers.upSampling2d({size:[factor,factor], batchInputShape:[1, w, h, 3]})
    
    return upsampler.apply(tf.expandDims(image, 0)).squeeze();
})}


/**
 * Pads an image tensor with 0s to be a square with sides that are powers of 2
 * @param   {tf.Tensor} image The unpadded image tensor
 * @returns {tf.Tensor}       The image, padded to the nearest power of 2         
 */
function padImage(image) {
    let [w, h, c] = image.shape

    let padded_size = getPaddedSize(image);
    return image.pad([[0, padded_size - w], [0, padded_size-h], [0,0]])
}


/**
 * Adds an alpha channel to a 3-channel image tensor
 * @param   {tf.Tensor} im            A 3-channel image tensor
 * @param   {Boolean}   disposeInputs Dispose the input image?
 * @returns {tf.Tensor}               The image with the alpha channel added
 */
function addAlpha(im, disposeInputs=true) {return tf.tidy(() => {
    let [h,w,c] = im.shape;
    let alpha   = tf.mul(tf.ones([h, w, 1]), 255);
    let result  = im.concat(alpha, -1);
    if (disposeInputs) {im.dispose();}
    return result;
})}


/**
 * KATANA PIPELINE
 */


/**
 * Returns an Array of LODs for an image and blendmode. The LODs are designed
 * so that you can get from one to the other by applying an inbetween in the correct blend mode.
 * @param   {tf.Tensor} image     The image tensor you want LODs of
 * @param   {String}    blendmode The blendmode to create the LODs for
 * @returns {Array}               The LODs of the image, created for the blendmode
 */
function generateLODs(image, blendmode) {
    let [w,h,c]    = image.shape;
    let num_layers = Math.ceil(Math.log2(h));
    let LODs       = [];

    // TODO: async this
    for (let i=0;i<num_layers; i++) {
        LODs.push(pixellate(image, i+1, blendmode));
    }

    return LODs;
}


/**
 * Generates the inbetween for 2 LODs in the correct blend mode
 * @param   {tf.Tensor} a         The first LOD, scaled to the same size as the second
 * @param   {tf.Tensor} b         The second LOD
 * @param   {String}    blendmode The blend mode being used
 * @returns                       The image tensor that, when added to a, with the selected blendmode, results in b
 */
function generateLODInbetween(a,b,blendmode) {return tf.tidy(() => {
    if (blendmode==='multiply') {
        return tf.div(b, tf.add(a, .001))
    } else if (blendmode==='screen') {
        return tf.sub(1, tf.div(tf.sub(1, b), tf.add(tf.sub(1, a), .001)))
    } else if (blendmode==='darken') {
        return tf.add(tf.mul(tf.greater(a, b), b), tf.mul(tf.greaterEqual(b, a), 1))
    } else if (blendmode==='lighten') {
        return tf.mul(tf.greater(b, a), b)
    } else if (blendmode==='plus-lighter') {
        return tf.sub(b, a)
    }
})}


//TODO: LODs to inbetweens separate fn?

/**
 * Generates LOD inbetweens (plus the base layer) for an image
 * @param   {tf.Tensor} image     The image you want inbetweens for
 * @param   {String}    blendmode The blendmode you are using
 * @returns {Promise<Array<tf.Tensor>>} A Promise that resolves to the inbetweens that generate the image
 */
function generateLODInbetweens(image, blendmode='screen', disposeInputs=true){

    let padded_image = padImage(image)
    let padded_size  = getPaddedSize(image);

    if(disposeInputs) {image.dispose();}

    let num_layers    = Math.ceil(Math.log2(padded_size/2))
    // await then
    let LODs          = generateLODs(padded_image, blendmode);

    let layerPromises = [new Promise((resolve, reject) => {resolve(tf.clone(LODs[0]));})]

    for (let i=0; i<num_layers;i++){layerPromises.push(new Promise((resolve, reject) => {resolve(tf.tidy(() => {
        // TODO: most of this all in generateinbetween
        let last_layer = scaleBy(LODs[i]);
        let new_image  = generateLODInbetween(last_layer, LODs[i+1], blendmode);

        let clippedImage = new_image.clipByValue(0, 1);
        return clippedImage;
    }))}))}

    return Promise.all(layerPromises).then(
        (layers) => new Promise(
            (resolve, reject) => {tf.dispose(LODs); tf.dispose(padded_image); resolve(layers)}
        )
    );
}


/**
 * Resizes and clips inbetweens to be the same shape as the original image
 * @param   {tf.Tensor}        image         The original image, used to get the clipped size
 * @param   {Array<tf.Tensor>} layers        The inbetweens generated from the original image
 * @param   {Boolean}          disposeInputs Dispose the inputs?
 * @returns {Array<tf.Tensor>}               The resized and clipped layers
 */
function cleanLODInbetweens(image, layers, disposeInputs=true) {
    let [w,h,c]       = image.shape;
    let padded_size   = getPaddedSize(image)
    let cleanPromises = []

    layers.forEach((layer) => {cleanPromises.push(new Promise((resolve, reject) => {resolve(tf.tidy(() => {
        
        let cleaned_layer = scaleBy(layer, Math.floor(padded_size/layer.shape[0]));
        let slicedLayer   = cleaned_layer.slice([0,0],[w, h]);

        if (disposeInputs) {layer.dispose();}

        return slicedLayer;
        // this is an atrocious mess of parentheses. :(

    }))}))})

    return Promise.all(cleanPromises);
}

// TODO: return a cloned copy and dispose optional
// avoid same name.
// here there was a problem where it would randomly select the same layer twice, resulting im memory leakage
function shuffleLayers(layers) {
    let [h,w,c] = layers[0].shape;
    for (let i=0; i<20; i++) {
        let indexA = Math.floor(Math.random()*layers.length);
        let indexB = Math.floor(Math.random()*(layers.length-1));
        if (indexB >=indexA) {indexB ++;}

        let [a,b] = returnSwapped(layers[indexA], layers[indexB]);

        layers[indexA].dispose();
        layers[indexB].dispose();
        layers[indexA] = a;
        layers[indexB] = b;
    }
    return layers;
}

function shuffleLayersAsync(layers) {
    let [h,w,c] = layers[0].shape;

    let indices = Array.from(
    { length: layers.length },
    (value, index) => index
    );

    let shufflePromises = [];

    while (indices.length >= 2) {
        let indexA = indices[Math.floor(Math.random()*indices.length)];
        indices.splice(indices.indexOf(indexA), 1)
        let indexB = indices[Math.floor(Math.random()*indices.length)];
        indices.splice(indices.indexOf(indexB), 1)

        shufflePromises.push(new Promise((resolve, reject) => {
            let [a,b] = returnSwapped(layers[indexA], layers[indexB]);
            layers[indexA].dispose();
            layers[indexB].dispose();
            resolve([a,b]);
        }))
    }

    if (indices.length == 1) {
        shufflePromises.push(new Promise((resolve, reject) => {
            resolve([tf.clone(layers[indices[0]])])
        }))
    }

    return Promise.all(shufflePromises).then(
        (shuffled) => new Promise(
            (resolve, reject) => {
                tf.dispose(layers);
                let unpacked = [];
                shuffled.forEach((shuffleSet) => {
                    shuffleSet.forEach((item) => {
                        unpacked.push(item);
                    })
                })
                resolve(unpacked);
            }
        )
    );
}

// promises here too, or in alike
// MAYBE THE SINGLE BASE LOD LAYER GOTTEN FROM LODS
// IS RESPONSIBLE FOR ALL OF THIS
function returnSwapped(a,b) {return tf.tidy(() => {
    let [h,w,c] = a.shape;

    let sliceX = Math.floor(Math.random()*w/2);
    let sliceY = Math.floor(Math.random()*h/2);
    let sliceH = Math.floor(h/2);
    let sliceW = Math.floor(w/2);

    let sliceA = tf.clone(a.slice([sliceY, sliceX],[sliceH, sliceW]));
    let sliceB = tf.clone(b.slice([sliceY, sliceX],[sliceH, sliceW]));

    let updateMapA = sliceB.pad([[sliceY, h-(sliceY+sliceH)],[sliceX, w-(sliceX+sliceW)],[0,0]],-1);
    let updateMapB = sliceA.pad([[sliceY, h-(sliceY+sliceH)],[sliceX, w-(sliceX+sliceW)],[0,0]],-1);

    return [a.where(tf.equal(updateMapA, -1), updateMapA), b.where(tf.equal(updateMapB, -1), updateMapB)];
})}

// Creates image box from images


function layers2KatanaBox(layers, blendmode, disposeInputs=true) {

    let boxPromises = []

    layers.forEach((layer) => {
        boxPromises.push(tensor2Img(layer));
    });

    return Promise.all(boxPromises).then((imgs) => {

        // will the document always be accessible?
        let katanaBox       = document.createElement('div');
        katanaBox.className = 'katana-box';

        imgs.forEach((img) => {
            styleKatanaImage(img, blendmode);
            katanaBox.appendChild(img);
        })

        return katanaBox;
    })
}


/**
 * Styles a katana image to have the correct positioning, class, and blendmode
 * @param   {Element} img The <img> element to style
 */
function styleKatanaImage(img, blendmode) {
    img.className = 'katana-image';
    img.style.position = 'absolute';
    img.style.mixBlendMode = blendmode;
}



/**
 * Turns an image into a Katana box, a div with the encrypted layers inside. The full pipeline in 1 function
 * @param {tf.Tensor} image     The (0-255) image tensor you want encrypted
 * @param {String}    blendmode The blendmode to use ('screen,'multiply','darken','lighten','plus-lighter')
 * @param {Boolean}   shuffle   Whether to shuffle sections of the layers around (highly recommended)
 */
function createKatanaBoxFromImage(image, blendmode='screen', shuffle=true) {

    console.log('katana box called', tf.memory());

    // a lot of this disposal is pre-done.

    return generateLODInbetweens(image, blendmode)
    .then((inbetweens) => {

        let t = Date.now();
        let result = cleanLODInbetweens(image, inbetweens);
        tf.dispose(inbetweens);
        console.log('clean ended, time ', (Date.now() - t) / 1000);
        return result;

    }).then((clean_layers) => {

        let t = Date.now();
        if (shuffle) {
            let r = shuffleLayersAsync(clean_layers);
            // TODO: wrong, r is a promise
            console.log('shuffle ended, time ', (Date.now() - t) / 1000);
            return r
        }
        else {return clean_layers;}

    }).then((shuffled_layers) => {

        let t = Date.now();
        let result = layers2KatanaBox(shuffled_layers, blendmode);
        //TODO: this in 2katanabox
        tf.dispose(shuffled_layers);
        console.log('make box ended, time ', (Date.now() - t) / 1000);
        return result;

        // the above still has the data to load async.

    });
}


/**
 * CONVERSION
 */


/**
 * Takes an image tensor and returns an <img> element
 * @param   {tf.Tensor} image The image tensor you want to be placed in an <img>
 * @returns                   An <img> element containing the image tensor
 */
function tensor2Img(image) {

    // <img> element expects an image with alpha
    // TODO: NEVER EVER REASSIGN
    let tensor = tf.mul(image, 255);
    if (tensor.shape[2]==3) {tensor = addAlpha(tensor);}
    let [h,w,c] = tensor.shape;

    return tensor.data().then((tensorData) => {
        let imageData = new ImageData(
        Uint8ClampedArray.from(tensorData),
        w, h);

        // The canvas stores the pixels, the <img>
        // takes the canvas's url and displays it.
        const canvas        = document.createElement('canvas');
              canvas.width  = w;
              canvas.height = h;
        const ctx           = canvas.getContext('2d');
              ctx.putImageData(imageData,0,0);

        // Create the image and return.
        let img = document.createElement('img')
        img.src = canvas.toDataURL('image/jpeg');

        tensor.dispose();

        return img;
    })
}



/**
 * Gets an image tensor from an image <input> element
 * @param   {string}    input The input element, which shoild contain a jpeg..
 * @returns {tf.Tensor}       A 3-channel, 0-1 tensor of shape [h,w,c]
 */
function imagesFromInputField(input) {

    let promises = [];
    
    Array.from(input.files).forEach((file) => promises.push(new Promise((resolve, reject) => {
        var reader = new FileReader();

        // "Do this once you load the file"
        reader.onloadend = (evt) => {
            let img = new Image();
            img.addEventListener('load', () => resolve(tf.browser.fromPixels(img)));
            img.addEventListener('error', (err) => reject(err));
            img.src = evt.target.result;
        };

        reader.readAsDataURL(file);
    })))

    console.log(promises);

    return Promise.all(promises)
}


/**
 * SETUP
 */


// remove tensors after 1/2 GB
tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 500000000);
// console.log("Using Backend ", tf.getBackend());