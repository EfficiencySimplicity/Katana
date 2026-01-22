
function getPaddedSize(image) {
    let [w, h, c] = image.shape
    return Math.ceil(2**Math.ceil(Math.log2(Math.max(w,h))))
}


function padImage(image) {
    let [w, h, c] = image.shape

    let padded_size = getPaddedSize(image);
    return image.pad([[0, padded_size - w], [0, padded_size-h], [0,0]])
}


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