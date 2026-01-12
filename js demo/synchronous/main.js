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


// Expects a 3d array of size (width, height, 3), and range 0-1.
function pixellate(image, depth=1, method='screen') {return tf.tidy(() => {

    let [w, h, c] = image.shape
    let d         = Math.floor(2 ** depth)

    // Reshape it into (d, d, chunk_size, chunk_size, 3)
    let pixel_map = image.reshape([d, Math.floor(w/d), d, Math.floor(h/d), c])
    pixel_map     = tf.einsum('abcde->acbde', pixel_map);

    // Each function (min/max) collapses each chunk into a single value,
    // resulting in shape (d,d,3)

    if (['multiply', 'darken', 'max'].includes(method)){
        pixel_map = pixel_map.max(axis = [2,3])
    }else if (['screen', 'lighten', 'plus-lighter', 'min'].includes(method)){
        pixel_map = pixel_map.min(axis = [2,3])
    }

    return pixel_map
})}


function scaleBy(image, factor=2) {return tf.tidy(() => {
    // Just create a whole layer object! What could go wrong?
    // Needless to day, clean this. Won't it always be 2?
    let [w, h, c] = image.shape
    let upsampler = tf.layers.upSampling2d({size:[factor,factor], batchInputShape:[1, w, h, 3]})
    
    return upsampler.apply(tf.expandDims(image, 0)).squeeze();
})}


function padImage(image) {
    let [w, h, c] = image.shape

    let next_power_of_2 = Math.ceil(2**Math.ceil(Math.log2(Math.max(w,h))))//???
    let padded_image    = image.pad([[0, next_power_of_2 - w], [0, next_power_of_2-h], [0,0]])

    return padded_image
}

// Accepts padded 0-1 images, generates LODs.
function generateLODs(image, method) {
    let [w,h,c]    = image.shape;
    let num_layers = Math.ceil(Math.log2(h));
    let LODs       = [];

    for (let i=0;i<num_layers; i++) {
        LODs.push(pixellate(image, i+1, method));
    }

    return LODs;
}


function LODstep(a,b,method) {return tf.tidy(() => {
    if (method==='multiply') {
        return tf.div(b, tf.add(a,.001))
    } else if (method==='screen') {
        return tf.sub(1, tf.div(tf.sub(1, b), tf.add(tf.sub(1,a),.001)))
    } else if (method==='darken') {
        return tf.add(tf.mul(tf.greater(a, b), b), tf.mul(tf.greaterEqual(b, a), 1))
    } else if (method==='lighten') {
        return tf.mul(tf.greater(b,a),b)
    } else if (method==='plus-lighter') {
        return tf.sub(b, a)
    }
})}


function generateKatanaLayers(image, method='screen'){
    
    let [ow,oh,oc]   = image.shape;
    let padded_image = padImage(image)
    let [w,h,c]      = padded_image.shape

    let num_layers    = Math.ceil(Math.log2(h/2))
    let LODs          = generateLODs(padded_image, method);
    let katana_layers = []

    // MAIN PIPELINE LOOP
    
    for (let i=0; i<num_layers;i++){
        let last_layer = scaleBy(LODs[i]);
        let new_image = LODstep(last_layer, LODs[i+1], method);
        last_layer.dispose();

        katana_layers.push(new_image.clipByValue(0, 1));
        new_image.dispose();

        console.log(tf.memory())
    }

    // FINAL POST PROCESSING AND EXPORT

    let cleaned_layers = []

    katana_layers.forEach((katana_layer) => {
        let cleaned_layer = scaleBy(katana_layer, Math.floor(w/katana_layer.shape[0]));
        cleaned_layers.push(cleaned_layer.slice([0,0],[ow, oh]))
        cleaned_layer.dispose();
        katana_layer.dispose();

        console.log(tf.memory())
    })

    return cleaned_layers;
}

// Creates image box from images
// This should *return* the box for later use
function layers2ImageBox(layers, blendmode) {
    let imageBox       = document.createElement('div');
    imageBox.className = 'image-box';

    // for now, place that box in the body
    let body = document.getElementById('body');
    body.appendChild(imageBox);

    layers.forEach((layer) => {
        // :M
        let img = arr2img(tf.mul(layer, 255));
        img.style.mixBlendMode = blendmode;
        imageBox.appendChild(img);
    })
}


// Adds alpha channel to RGB image, 
// necessary for making an <img> element
function addAlpha(arr) {return tf.tidy(() => {
    let alpha = tf.mul(tf.ones([arr.shape[0], arr.shape[1], 1]), 255);
    arr       = arr.concat(alpha, -1);
    return arr;
})}


// Takes in a tensor and returns an <img> element.
// Assumes values ranging from 0 to 255.
function arr2img(arr) {
    // Image element expects alpha
    // Tensor and arr are bad names. both are tensors,
    // one just has alpha.
    let tensor = arr;
    if (arr.shape[2]==3) {tensor = addAlpha(arr);}
    let [h,w,c] = tensor.shape;

    // Remove dataSync in async version?
    let data = new ImageData(
        Uint8ClampedArray.from(tensor.dataSync()),
        w, h);

    // The canvas stores the pixels, the image
    // takes the canvas's url and displays it.
    const canvas   = document.createElement('canvas');
    canvas.width   = w;
    canvas.height  = h;
    const ctx      = canvas.getContext('2d');
    ctx.putImageData(data,0,0);

    // Create the image and return.
    let img = document.createElement('img')
    img.src = canvas.toDataURL('image/jpeg');
    img.className = 'move-image';
    return img;
}

function getCurrentBlendModeSelected() {
    // https://stackoverflow.com/questions/1085801/get-selected-value-in-dropdown-list-using-javascript
    return document.getElementById("blend-mode-select").value;
}

// Arr is assumed to have 3 channels, and have values ranging from 0 to 255.
function katanifyAndInsertInDocument(arr) {
    // width and height seem to be flipped, but
    // that won't affect anything.
    // just note that in the above funstions,
    // w stands for height, and h stands for width
    // change it later.

    //:M
    arr = tf.div(arr, 255.0);

    let blendmode = getCurrentBlendModeSelected();//'screen';

    let layers    = generateKatanaLayers(arr, blendmode)
    
    layers2ImageBox(layers, blendmode);
}


// Gets image array from file imput, demo-specific.
// https://stackoverflow.com/questions/35274934/retrieve-image-data-from-file-input-without-a-server
function createImageBoxFromInputFile() {
    let existingImage = document.querySelector('.image-box');
    if (existingImage) {existingImage.remove();}

    var file   = document.getElementById('file-input').files[0];
    var reader = new FileReader();

    // Array.from to get multiple files from the input's FileList
    // or const file of files

    // "Do this once you load the file"
    reader.onloadend = (evt) => {
        let img = new Image();

        img.onload = () => {
            let tensor = tf.browser.fromPixels(img);
            // tf tidy clears up all the layers after creating image.
            // Intermediate tensors are handled inside this function.
            tf.tidy(() => katanifyAndInsertInDocument(tensor));
            tensor.dispose();
            // Should have 0 tensors
            console.log('Post-operation memory:', tf.memory());
        }

        img.src = evt.target.result;
    };

    // "'kay now start loading the file"
    if (file) {reader.readAsDataURL(file);}
}


// remove tensors after 1/2 Gb
tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 500000000);