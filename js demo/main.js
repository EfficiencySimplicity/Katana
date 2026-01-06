// JS DEMO
// 
// Normal mode is omitted, as it is both needlessly complex and performs worse
// than all other modes.


// image is a 3d array of size (width, height, 3)
function pixellate_at_depth(image, depth=1, method='screen') {
    let d = Math.floor(2 ** depth)
    let [w, h, c] = image.shape

    // Reshape it into (d, d, chunk_size, chunk_size, 3)
    // Translate from numpy

    // let pixel_map = image.reshape(d, Math.floor(w/d), d, Math.floor(h/d), c)
    // pixel_map = pixel_map.transpose(0,2,1,3,4)

    // numpy functions should be replaced, and
    // each function (min/max) should collapse each chunk into a single value,
    // resulting in shape (d,d,3)

    if (['multiply', 'darken'].includes(method)){
        //pixel_map = np.max(pixel_map, axis = (2,3))
    }else if (['screen', 'lighten', 'pluslighter'].includes(method)){
        //pixel_map = np.min(pixel_map, axis = (2,3))
    }

    return pixel_map
}

// This needs to upscale a pixellated image, so that it look sexacly the same, but each pixel is now
// factor*factor pixels in the new image

// function scale_by(image, factor=2) {
//     image = np.repeat(image, factor, axis=0)
//     image = np.repeat(image, factor, axis=1)
//     return image
// }

// This could be simplified if your module has a specified pad() function
function get_padded_image(image) {

    let [w, h, c] = image.shape

    // Nah, I'm not translating all this!

    // let next_power_of_2 = int(2**np.ceil(np.log2(max(list(image.shape)[ {2]))))
    // let padded_image = np.zeros((next_power_of_2, next_power_of_2, 3))
    // padded_image[ {w,  {h] = image

    return padded_image
}

// these should be the easiest to translate

// function get_next_layer_multiply(target_map, current_map) {
//     let new_image = (target_map / (current_map+.001))
//     current_map = current_map * new_image

//     return current_map, new_image
// }

// function get_next_layer_screen(target_map, current_map) {
//     let new_image = 1 - ((1-target_map) / ((1-current_map)+.001))
//     current_map = 1 - (1-current_map)*(1-new_image)

//     return current_map, new_image
// }

// function get_next_layer_screen(target_map, current_map) {
//     let new_image = 1 - ((1-target_map) / ((1-current_map)+.001))
//     current_map = 1 - (1-current_map)*(1-new_image)

//     return current_map, new_image
// }

// function get_next_layer_lighten(target_map, current_map) {
//     let new_image = (target_map > current_map) * target_map
//     current_map = np.maximum(current_map, new_image)

//     return current_map, new_image
// }

// function get_next_layer_plus_lighter(target_map, current_map) {
//     let new_image = target_map - current_map
//     current_map = current_map + new_image

//     return current_map, new_image
// }

function generate_component_images(image, method='screen'){
    
    let padded_image = get_padded_image(image)
    let [w,h,c] = padded_image.shape

    let base = pixellate_at_depth(padded_image, starting_depth, method)

    // let current_map = np.array(base)
    // let new_image // We'll be using this in the loop

    // num_layers = int(
    //     np.log2(
    //         h / base.shape[0]
    //     )
    // )

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
        }else if (method === 'pluslighter'){
            [current_map, new_image] = get_next_layer_plus_lighter(target_map, current_map)
        }

        // new_image = np.clip(new_image, 0, 1)
        new_image = scale_by(new_image, Math.floor(w/new_image.shape[0]))
        layer_images.push(new_image)
    }

    // FINAL POST PROCESSING AND EXPORT

    base = scale_by(base, Math.floor(w/base.shape[0]))

    // resize to non-padded size
    //base = base[:image.shape[0], :image.shape[1]]

    let clipped_layers = []

    layer_images.forEach((layer_image, idx) => {
        //clipped_layers.push(layer_image[:image.shape[0], :image.shape[1]])
    })

    return [base, clipped_layers]
}

// function to get image pixel data from the input box that will exist in future


// function that takes in an <img> and an array and puts the array in the <img>


// function that changes blend mode of <imgs> when dropdown changed