// JS DEMO
// 
// This thing is held together by nothing but tape
// , paper-mache, and hope. Tread carefully.
//
// MEMORY LEAKS TO BE FIXED LATER THIS IS PROOF OF CONCEPT
// Memory leaks are less, at least, but still slowly exist.
//
// Normal mode is omitted, as it is both needlessly complex and performs worse
// than all other modes.
//
// The naming on some of these functions is atrocious.
// I am sorry.
//
// Plus some old comments from pre-translation are left


// image is a 3d array of size (width, height, 3)
function pixellate_at_depth(image, depth=1, method='screen') {
    let d = Math.floor(2 ** depth)
    let [w, h, c] = image.shape

    // Reshape it into (d, d, chunk_size, chunk_size, 3)
    let pixel_map = image.reshape([d, Math.floor(w/d), d, Math.floor(h/d), c])
    pixel_map = tf.einsum('abcde->acbde', pixel_map);

    // each function (min/max) should collapse each chunk into a single value,
    // resulting in shape (d,d,3)

    if (['multiply', 'darken'].includes(method)){
        pixel_map = pixel_map.max(axis = [2,3])
    }else if (['screen', 'lighten', 'plus-lighter'].includes(method)){
        pixel_map = pixel_map.min(axis = [2,3])
    }

    console.log(pixel_map.shape)

    return pixel_map
}

// This needs to upscale a pixellated image, so that it look sexacly the same, but each pixel is now
// factor*factor pixels in the new image

function scale_by(image, factor=2) {
    // just create a whole layer object! What could go wrong?
    let lay = tf.layers.upSampling2d({size:[factor,factor], batchInputShape:[1, image.shape[0], image.shape[1], 3]})
    let result = lay.apply(tf.expandDims(image, 0)).squeeze();
    console.log(result.shape);
    return result
}

// This could be simplified if your module has a specified pad() function
function get_padded_image(image) {

    let [w, h, c] = image.shape

    let next_power_of_2 = Math.ceil(2**Math.ceil(Math.log2(Math.max(w,h))))
    console.log(next_power_of_2);
    let padded_image = image.pad([[0, next_power_of_2 - w], [0, next_power_of_2-h], [0,0]])
    console.log(padded_image.shape);
    return padded_image
}

// these should be the easiest to translate

function get_next_layer_multiply(target_map, current_map) {
    let new_image = tf.div(target_map, tf.add(current_map,.001))
    current_map = tf.mul(current_map, new_image)

    return [current_map, new_image]
}

function get_next_layer_screen(target_map, current_map) {
    let new_image = tf.sub(1, tf.div(tf.sub(1, target_map), tf.add(tf.sub(1,current_map),.001)))
    current_map = tf.sub(1,tf.mul(tf.sub(1,current_map),tf.sub(1,new_image)))

    return [current_map, new_image]
}

function get_next_layer_darken(target_map, current_map) {
    let new_image = tf.add(tf.mul(tf.greater(current_map, target_map), target_map), tf.mul(tf.greaterEqual(target_map, current_map), 1))
    current_map = tf.minimum(current_map, new_image)

    return [current_map, new_image]
}

function get_next_layer_lighten(target_map, current_map) {
    let new_image = tf.mul(tf.greater(target_map,current_map),target_map)
    current_map = tf.maximum(current_map, new_image)

    return [current_map, new_image]
}

function get_next_layer_plus_lighter(target_map, current_map) {
    let new_image = tf.sub(target_map, current_map)
    current_map = tf.add(current_map,new_image)

    return [current_map, new_image]
}

function generate_component_images(image, method='screen'){
    
    let [ow,oh,oc] = image.shape;
    let padded_image = get_padded_image(image)
    let [w,h,c] = padded_image.shape

    let base = pixellate_at_depth(padded_image,starting_depth=1,method)

    let current_map = tf.clone(base)
    let new_image // We'll be using this in the loop

    num_layers = Math.ceil(
        Math.log2(
            h / base.shape[0]
        )
    )

    let layer_images = []

    // MAIN PIPELINE LOOP
    
    for (let i=0; i<num_layers;i++){

        let this_depth = starting_depth + i + 1

        let target_map = pixellate_at_depth(padded_image, this_depth, method = method)

        current_map = scale_by(current_map)

        if (method === 'multiply'){
            [current_map, new_image] = get_next_layer_multiply(target_map, current_map)
        } else if (method === 'screen'){
            [current_map, new_image] = get_next_layer_screen(target_map, current_map)
        } else if (method === 'lighten'){
            [current_map, new_image] = get_next_layer_lighten(target_map, current_map)
        } else if (method === 'darken'){
            [current_map, new_image] = get_next_layer_darken(target_map, current_map)
        }else if (method === 'plus-lighter'){
            [current_map, new_image] = get_next_layer_plus_lighter(target_map, current_map)
        }

        new_image = new_image.clipByValue(0, 1);
        new_image = scale_by(new_image, Math.floor(w/new_image.shape[0]));
        layer_images.push(new_image);
    }

    // FINAL POST PROCESSING AND EXPORT

    base = scale_by(base, Math.floor(w/base.shape[0]));

    // resize to non-padded size
    base = base.slice([0,0],[ow, oh])

    let clipped_layers = [base]

    layer_images.forEach((layer_image, idx) => {
        clipped_layers.push(layer_image.slice([0,0],[ow, oh]))
    })

    return clipped_layers
}

// function that creates image box from images

function createImageBoxFromLayers(layers, blendmode) {
    let body = document.getElementById('body');
    let imageBox = document.createElement('div');
    imageBox.className = 'image-box';
    body.appendChild(imageBox);

    layers.forEach((layer) => {
        let img = imageElementFromArray(tf.mul(layer, 255));
        img.style.mixBlendMode=blendmode;
        // place the layer matrix into the image
        imageBox.appendChild(img);
    })
}

// function to get image array from file imput
// https://stackoverflow.com/questions/35274934/retrieve-image-data-from-file-input-without-a-server
function createImageBoxFromInputFile() {
    let existingImage = document.querySelector('.image-box');
    if (existingImage) {
        existingImage.remove();
    }
    var file   = document.getElementById('file-input').files[0];
    var reader = new FileReader();

    // Array.from to get multiple files from the input's FileList
    // or const file of files

    reader.onloadend = (evt) => {
        let img = new Image();
        img.onload = () => {
            let tensor = tf.browser.fromPixels(img);
            // tf tidy turns 231 tensors into just 1. WOW
            // and 1078395816 bytes into 9830400, 107 times less.
            tf.tidy(() => createImageBoxFromOriginalImageArray(tensor));
            console.log('mem:', tf.memory());
        }
        img.src = evt.target.result;
    };

    if (file) {
        reader.readAsDataURL(file);
    }
}

// arr is assumed un normalized
function createImageBoxFromOriginalImageArray(arr) {
    // width and height seem to be flipped, but
    // that won't affect anything.
    // just note that in the above funstions,
    // w stands for height, and h stands for width

    // or is this so? .shape never been got.

    arr = tf.div(arr, 255.0);

    let blendmode = getCurrentBlendModeSelected();//'screen';

    let layers = generate_component_images(arr, blendmode)
    
    createImageBoxFromLayers(layers, blendmode);
}

// function to add alpha layer to image
function addAlpha(arr) {
    let alpha = tf.mul(tf.ones([arr.shape[0], arr.shape[1], 1]), 255);
    arr = arr.concat(alpha, -1);
    return arr;
}

// does what it says, assumes 255-scaled
function imageElementFromArray(arr) {
    let tensor = addAlpha(arr);
    // "We also provide synchronous versions of
    //  these methods which are simpler to use,
    //  but will cause performance issues 
    //  in your application. 
    // You should always prefer the asynchronous methods 
    // in production applications. ""
    let data = new ImageData(Uint8ClampedArray.from(tensor.dataSync()), tensor.shape[1], tensor.shape[0]);

    const c=document.createElement('canvas');
    c.width=tensor.shape[1];
    c.height=tensor.shape[0];
    const ctx=c.getContext('2d');
    ctx.putImageData(data,0,0);

    // document.body.appendChild(c);


    let img = document.createElement('img')
    img.src=c.toDataURL('image/jpeg');
    img.className = 'move-image';
    return img;
}

function getCurrentBlendModeSelected() {
    // https://stackoverflow.com/questions/1085801/get-selected-value-in-dropdown-list-using-javascript
    var e = document.getElementById("blend-mode-select");
    return e.value;
}

// // create dummy box
// document.getElementById('body').appendChild(
//     imageElementFromArray(
//         scale_by(
//             pixellate_at_depth(
//                 get_padded_image(tf.randomUniform([100,100,3], 0, 255)), 5
//             ), 2
//         )
//     )
// );
// console.log(tf.memory())