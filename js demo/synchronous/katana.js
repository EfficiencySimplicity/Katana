// JS DEMO
//
// Normal mode is omitted, as it is both needlessly complex and performs worse
// than all other modes.

// :M means memory is used up here. It is cleared later, but could be more efficient

// "We also provide synchronous versions of
//  these methods which are simpler to use,
//  but will cause performance issues 
//  in your application. 
// You should always prefer the asynchronous methods 
// in production applications. ""
// meaning all of this could be much faster if you use Promises

// TODO: use a max_layers arg.

//https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/log
// function getBaseLog(x, y) {
//   return Math.log(y) / Math.log(x);
// }


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
    if (['multiply', 'darken', 'max'].includes(blendmode)){
        return pixel_map.max(axis = [2,3])
    }else if (['screen', 'lighten', 'plus-lighter', 'min'].includes(blendmode)){
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

// TODO: js does it have an actual blendmode object?... or is that CSS?

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
        return tf.div(b, tf.add(a,.001))
    } else if (blendmode==='screen') {
        return tf.sub(1, tf.div(tf.sub(1, b), tf.add(tf.sub(1,a),.001)))
    } else if (blendmode==='darken') {
        return tf.add(tf.mul(tf.greater(a, b), b), tf.mul(tf.greaterEqual(b, a), 1))
    } else if (blendmode==='lighten') {
        return tf.mul(tf.greater(b,a),b)
    } else if (blendmode==='plus-lighter') {
        return tf.sub(b, a)
    }
})}


function generateLODInbetweens(image, blendmode='screen'){

    let padded_image = padImage(image)
    let padded_size  = padded_image.shape[0]

    let num_layers    = Math.ceil(Math.log2(padded_size/2))
    let LODs          = generateLODs(padded_image, blendmode);
    let katana_layers = [LODs[0]]

    // MAIN PIPELINE LOOP
    
    for (let i=0; i<num_layers;i++){
        let last_layer = scaleBy(LODs[i]);
        let new_image = generateLODInbetween(last_layer, LODs[i+1], blendmode);
        last_layer.dispose();

        katana_layers.push(new_image.clipByValue(0, 1));
        new_image.dispose();
    }

    return katana_layers;
}


function cleanLODInbetweens(image, layers) {return tf.tidy(() => {
    let [w,h,c]  = image.shape;
    let padded_size = getPaddedSize(image)
    let cleaned_layers = []

    layers.forEach((layer) => {
        let cleaned_layer = scaleBy(layer, Math.floor(padded_size/layer.shape[0]));
        cleaned_layers.push(cleaned_layer.slice([0,0],[w, h]))

        cleaned_layer.dispose();
        layer.dispose();
    })

    return cleaned_layers;
})}


function shuffleLayers(layers) {
    let [h,w,c] = layers[0].shape;
    for (let i=0; i<20; i++) {
        let indexA = Math.floor(Math.random()*layers.length);
        let indexB = Math.floor(Math.random()*layers.length);

        let [a,b] = returnSwapped(layers[indexA], layers[indexB]);
        layers[indexA].dispose();
        layers[indexB].dispose();
        layers[indexA] = a;
        layers[indexB] = b;
    }
    return layers;
}

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
// This should *return* the box for later use
function layers2KatanaBox(layers, blendmode) {return tf.tidy(() => {
    // will the document always be accessible?
    let katanaBox       = document.createElement('div');
    katanaBox.className = 'katana-box';

    layers.forEach((layer) => {
        let img = tensor2Img(layer);
        img.style.mixBlendMode = blendmode;
        katanaBox.appendChild(img);
    })

    return katanaBox;
})}


// Expects a 3-channel 0-255 tensor
// Returns a 4-channel 0-255 tensor, with the new channel filled with 255s
function addAlpha(im) {return tf.tidy(() => {
    let [h,w,c] = im.shape;
    let alpha   = tf.mul(tf.ones([h, w, 1]), 255);
    return im.concat(alpha, -1);
})}


/**
 * Takes an image tensor and returns an <img> element
 * @param   {tf.Tensor} image The image tensor you want to be placed in an <img>
 * @returns                   An <img> element containing the image tensor
 */
function tensor2Img(image) {return tf.tidy(() => {

    // <img> element expects an image with alpha
    let tensor = tf.mul(image, 255);
    if (tensor.shape[2]==3) {tensor = addAlpha(tensor);}
    let [h,w,c] = tensor.shape;

    // Remove dataSync in async version?
    // tf.browser.toPixels?
    let data = new ImageData(
        Uint8ClampedArray.from(tensor.dataSync()),
        w, h);

    // The canvas stores the pixels, the <img>
    // takes the canvas's url and displays it.
    const canvas        = document.createElement('canvas');
          canvas.width  = w;
          canvas.height = h;
    const ctx           = canvas.getContext('2d');
          ctx.putImageData(data,0,0);

    // Create the image and return.
    let img       = document.createElement('img')
    img.src       = canvas.toDataURL('image/jpeg');
    img           = styleKatanaImage(img);
    return img;
})}

// blend-mode later?


/**
 * Styles a katana image to have the correct positioning;
 * @param   {Element} img The <img> element to style
 * @returns               The element with basic Katana Image styling applied
 */
function styleKatanaImage(img) {
    img.className = 'katana-image';
    img.style.position = 'absolute';
    return img;
}


/**
 * Turns an image into a Katana box, a div with the encrypted layers inside.
 * @param {tf.Tensor} image     The (0-255) image tensor you want encrypted
 * @param {String}    blendmode The blendmode to use ('screen,'multiply','darken','lighten','plus-lighter')
 * @param {Boolean}   shuffle   Whether to shuffle sections of the layers around (highly recommended)
 */
function createKatanaBoxFromImage(image, blendmode='screen', shuffle=true) {
    image = tf.div(image, 255.0);

    let t = Date.now();

    let layers = generateLODInbetweens(image, blendmode)

    layers = cleanLODInbetweens(image, layers);

    console.log('clean ended, time ', (Date.now() - t) / 1000);
    t = Date.now();

    if (shuffle) {layers = shuffleLayers(layers);}

    console.log('shuffle ended, time ', (Date.now() - t) / 1000);
    t = Date.now();
    
    return layers2KatanaBox(layers, blendmode);
}


/**
 * Gets an image tensor from an image <input> element
 * @param   {string}    input The input element, which shoild contain a jpeg..
 * @returns {tf.Tensor}       A 3-channel, 0-255 tensor of shape [h,w,c]
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

// remove tensors after 1/2 GB
tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 500000000);