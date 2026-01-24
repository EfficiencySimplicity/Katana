// JS DEMO



/**
 * BASICS
 */



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
 * Pads an image tensor with 0s to be a square with sides that are powers of 2
 * @param   {tf.Tensor} image The unpadded image tensor
 * @returns {tf.Tensor}       The image, padded to the nearest power of 2         
 */
function padImage(image) {
    let [w, h, c]   = image.shape
    let padded_size = getPaddedSize(image);
    return image.pad([[0, padded_size - w], [0, padded_size-h], [0,0]])
}


/**
 * Pixellates an image tensor with the correct depth and blendmode
 * @param   {tf.Tensor} image     A square, 3-channel, 0-1 tensor representing an image
 * @param   {Number}    depth     The depth to pixellate to. The size of the result is 2^depth
 * @param   {String}    blendmode The blendmode used for pixellation
 * @returns {tf.Tensor}           The scaled-down tensor
 */
function pixellate(image, depth=1, blendmode='screen') {return tf.tidy(() => {

    let [w, h, c] = image.shape
    let d         = Math.floor(2 ** depth)

    // Reshape it into (d, d, chunk_size, chunk_size, 3)
    let pixel_map = image.reshape([Math.floor(w/d), d, Math.floor(h/d), d, c])
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
 * Calculates how much to scale a tensor for its shape to match another
 * @param   {tf.Tensor} a The tensor you are goint to scale
 * @param   {tf.Tensor} b A tensor with the target shape
 * @returns {Number}      What A must be scaled by to be the same size as b
 */
function getScaleRatio(a, b) {
    // TODO: error if not equal ratios on both sides
    return Math.floor(b.shape[0] / a.shape[0])
}


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
 * Adds an alpha channel to a 3-channel image tensor
 * @param   {tf.Tensor} image         A 3-channel image tensor
 * @param   {Boolean}   disposeInputs Dispose the input image?
 * @returns {tf.Tensor}               The image with the alpha channel added
 */
function addAlpha(image, disposeInputs=true) {return tf.tidy(() => {
    let [h,w,c] = image.shape;
    let alpha   = tf.mul(tf.ones([h, w, 1]), 255);
    let result  = image.concat(alpha, -1);
    if (disposeInputs) {image.dispose();}
    return result;
})}



/**
 * KATANA PIPELINE
 */



/**
 * Returns an Array of LODs for an image and blendmode. The LODs are designed
 * so that you can get from one to the other by applying an inbetween in the correct blendmode.
 * @param   {tf.Tensor} image     The image tensor you want LODs of
 * @param   {String}    blendmode The blendmode to create the LODs for
 * @param   {Number}    ratio     The ratio of one layer to the next, in powers of 2. Best at 2-3
 * @returns {Array}               The LODs of the image, created for the blendmode
 */
function generateLODs(image, blendmode, ratio=3) {
    let [w,h,c]    = image.shape;
    let num_layers = Math.ceil(Math.log2(h));
    let LODs       = [];

    // TODO: async this
    for (let i=0; i<=num_layers; i+=ratio) {
        LODs.unshift(pixellate(image, i, blendmode));
    }

    return LODs;
}


/**
 * Generates the inbetween for 2 LODs in the correct blendmode
 * @param   {tf.Tensor} a         The first LOD, scaled to the same size as the second
 * @param   {tf.Tensor} b         The second LOD
 * @param   {String}    blendmode The blendmode being used
 * @returns                       The image tensor that, when added to a, with the selected blendmode, results in b
 */
function generateLODInbetween(a, b, blendmode) {return tf.tidy(() => {
    let i;
    if (blendmode==='multiply') {
        i = tf.div(b, tf.add(a,.001))
    } else if (blendmode==='screen') {
        i = tf.sub(1, tf.div(tf.sub(1, b), tf.add(tf.sub(1, a), .001)))
    } else if (blendmode==='darken') {
        i = tf.add(tf.mul(tf.greater(a, b), b), tf.mul(tf.greaterEqual(b, a), 1))
    } else if (blendmode==='lighten') {
        i = tf.mul(tf.greater(b, a), b)
    } else if (blendmode==='plus-lighter') {
        i = tf.sub(b, a)
    }
    return i.clipByValue(0, 1)
})}


/**
 * Generates inbetweens for an array of LODs
 * @param   {Array<tf.Tensor>} LODs      The array of LODs you want inbetweens for
 * @param   {String}           blendmode The blendmode you're using
 * @returns {Array<tf.Tensor>}           The inbetweens that create the LODs
 */
function generateLODInbetweens(LODs, blendmode='screen', disposeInputs=true){

    let inbetweens = [tf.clone(LODs[0])];

    for (let i=0; i<LODs.length - 1; i++){
        let a = scaleBy(LODs[i], getScaleRatio(LODs[i], LODs[i+1]));
        let b = LODs[i+1];

        inbetweens.push(generateLODInbetween(a, b, blendmode))

        a.dispose();
    }

    if (disposeInputs) {tf.dispose(LODs)}

    return inbetweens;
}


/**
 * Resizes and clips inbetweens to the same shape as the original image
 * @param   {tf.Tensor}        image      The original image, used as a reference
 * @param   {Array<tf.Tensor>} inbetweens The inbetweens to clean
 * @returns {Array<tf.Tensor>}            The cleaned layers
 */
function cleanLODInbetweens(image, inbetweens) {return tf.tidy(() => {
    let [w,h,c]            = image.shape;
    let padded_size        = getPaddedSize(image)
    let cleaned_inbetweens = []

    inbetweens.forEach((image) => {

        let scale_factor = Math.floor(padded_size/image.shape[0]); //TODO: size ratio somehow?
        let scaled_image = scaleBy(image, scale_factor);
        let sliced_image = scaled_image.slice([0,0],[w, h])

        cleaned_inbetweens.push(sliced_image);

        scaled_image.dispose();
        image.dispose();
    })

    return cleaned_inbetweens;
})}


// TODO: return a cloned copy and dispose optional
// and shuffle at a lower res, for goodness sakes.
// avoid same name.
// here there was a problem where it would randomly select the same layer twice, resulting im memory leakage

/**
 * Shuffles slices of an array of tensors between each other
 * @param   {Array<tf.Tensor>} layers   The tensors to shuffle
 * @param   {Number}           shuffles The number of shuffles to perform
 * @returns {Array<tf.Tensor>}          An array of shuffled tensors
 */
function shuffleLayers(layers, shuffles = 20) {
    for (let i=0; i<shuffles; i++) {
        let indexA = Math.floor(Math.random()*layers.length);
        let indexB = Math.floor(Math.random()*(layers.length-1));
        if (indexB >=indexA) {indexB ++;}

        let [a,b] = shuffleTwoLayers(layers[indexA], layers[indexB]);

        layers[indexA].dispose();
        layers[indexB].dispose();
        layers[indexA] = a;
        layers[indexB] = b;
    }
    return layers;
}


/**
 * Shuffles a slice of 2 tensors between each other
 * @param   {tf.Tensor} a      The first layer
 * @param   {tf.Tensor} b      The second layer
 * @returns {Array<tf.Tensor>} The shuffled layers
 */
function shuffleTwoLayers(a, b) {return tf.tidy(() => {
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


/**
 * Creates a Katana box from an array of image tensors
 * @param   {Array<tf.Tensor>} layers    The layers to make into a box
 * @param   {String}           blendmode The blendmode to use
 * @returns 
 */
function layers2KatanaBox(layers, blendmode) {
    // do  I need to tidy?!
    let katanaBox       = document.createElement('div');
    katanaBox.className = 'katana-box';

    layers.forEach(async (layer) => {
        let img = await tensor2Img(layer);
        styleKatanaImage(img, blendmode);
        katanaBox.appendChild(img);
    })

    // TODO: mesh this in with async and set proper size
    // katanaBox.firstChild.style.mixBlendMode = 'normal';

    return katanaBox;
}


/**
 * Styles a katana image to have the correct positioning and blendmode
 * @param   {Element} img       The <img> element to style
 * @param   {String}  blendmode The blendmode to apply
 * @returns                     The element with basic Katana Image styling applied
 */
function styleKatanaImage(img, blendmode) {
    img.className = 'katana-image';
    img.style.position = 'absolute';
    img.style.mixBlendMode = blendmode;
}


/**
 * Turns an image into a Katana box, a div with the encrypted layers inside.
 * @param {tf.Tensor} image     The (0-255) image tensor you want encrypted
 * @param {String}    blendmode The blendmode to use ('screen,'multiply','darken','lighten','plus-lighter')
 * @param {Boolean}   shuffle   Whether to shuffle sections of the layers around (highly recommended)
 */
function createKatanaBoxFromImage(image, blendmode='screen', shuffles=0, layer_ratio=1) {

    let t = Date.now();

    console.log('B4 padding: ', tf.memory().numTensors);

    let padded_image = padImage(image);

    console.log('AT padding, B4 LODs: ', tf.memory().numTensors);

    let LODs = generateLODs(padded_image, blendmode, layer_ratio);

    console.log('AT LODs, B4 IBTs: ', tf.memory().numTensors);

    let layers = generateLODInbetweens(LODs, blendmode)

    console.log('AT IBTs, B4 cleaning: ', tf.memory().numTensors);

    layers = cleanLODInbetweens(image, layers);

    console.log('AT cleaning, B4 shuffle: ', tf.memory().numTensors);

    layers = shuffleLayers(layers, shuffles);
    
    console.log('AT shuffle, B4 box making: ', tf.memory().numTensors);

    console.log('Full time taken before making box: ', (Date.now() - t) / 1000);
    
    return layers2KatanaBox(layers, blendmode);
}



/**
 * CONVERSION
 */



/**
 * Converts an image tensor to an <img> element
 * @param   {tf.Tensor}        image The image tensor you want to be placed in an <img>
 * @returns {Promise<Element>}       A promise that resolves to an <img> containing the image tensor
 */
function tensor2Img(image) {

    // <img> element expects an image with alpha
    // TODO: NEVER EVER REASSIGN
    let tensor = tf.mul(image, 255);
    if (tensor.shape[2]==3) {tensor = addAlpha(tensor);}
    let [h,w,c] = tensor.shape;

    return tensor2array(tensor).then((tensorData) => {
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

        // Create the image and return
        let img = document.createElement('img')
        img.src = canvas.toDataURL('image/jpeg');

        return img;
    })
}


function tensor2array(tensor, disposeInputs=true) {
    return tensor.data().then((data) => {if (disposeInputs) {tensor.dispose();} return data;});
}


function array2tensor(array, disposeInputs=true) {
    return tensor.data().then((data) => {if (disposeInputs) {tensor.dispose();} return data;});
}


// NOTE: HighDuck404 corrected 'shoild' to 'should'
/**
 * Gets image tensors from an image <input> element
 * @param   {string}    input           The input element, which should contain a jpeg or webp..
 * @returns {Promise<Array<tf.Tensor>>} Resolves to an array of 3-channel, 0-255 tensors of shape [h,w,c]
 */
function imagesFromInputField(input) {
    let promises = [];
    
    Array.from(input.files).forEach((file) => promises.push(imageFromInputFile(file)))

    return Promise.all(promises)
}


/**
 * Gets an image tensor from an <input> element's file
 * @param   {string}    file An input element's file, which should be a jpeg or webp..
 * @returns {tf.Tensor}      A 3-channel, 0-255 tensor of shape [h,w,c]
 */
function imageFromInputFile(file) {return new Promise((resolve, reject) => {
    var reader = new FileReader();

    // "Do this once you load the file"
    reader.onloadend = (evt) => {
        let img = new Image();
        img.addEventListener('load', () => resolve(tensorFromLoadedImage(img)));
        img.addEventListener('error', (err) => reject(err));
        img.src = evt.target.result;
    };

    // "Load the file"
    reader.readAsDataURL(file);
})}


/**
 * Gets an image tensor from a loaded Image()
 * @param   {string}    image A loaded Image()
 * @returns {tf.Tensor}       A 3-channel, 0-255 tensor of shape [h,w,c]
 */
function tensorFromLoadedImage(image) {return tf.tidy(() => {
    let pixel_values = tf.browser.fromPixels(image);
    return tf.div(pixel_values, 255.0);
})}



/**
 * CONFIGURATION
 */



// remove tensors after 1/2 GB
tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 500000000);

// https://www.tensorflow.org/js/guide/platform_environment

// In the TensorFlow Node.js backend, 'node', the TensorFlow C API is used to accelerate operations. 
// This will use the machine's available hardware acceleration, like CUDA, if available.

tf.setBackend("webgl")

// tf.enableProdMode();

// tf.enableDebugMode();