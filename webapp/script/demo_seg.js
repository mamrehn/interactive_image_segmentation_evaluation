"use strict";

function setCookie(key, value, days) {
    let expires = '';
    if (days) {
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = '; expires=' + date.toUTCString();
    }
    document.cookie = key + '=' + value + expires + '; path=/';
}

function getCookie(key) {
    const name_eq = key + '=',
        cookie_parts = document.cookie.split(';'),
        len = cookie_parts.length;
    for(let i = 0; i < len; ++i) {
        let current_info = cookie_parts[i].trim();
        if (0 === current_info.indexOf(name_eq))
            return current_info.substring(name_eq.length, current_info.length);
    }
    return null;
}

// function delCookie(key) {setCookie(key, '', -1);}

const data_set_name_global = window.location.href.split('@')[1],
    user_name_global = window.location.href.split('@')[0].split('#')[1].replace(/,/g, '_'),
    overall_image_data_global = get_overall_image_data_global(data_set_name_global),
    test_result_global = {},
    contour_thickness_vals_global = [1/2, 1/3, 1/4, 0|0, 1|0],
    fg_seed_color_global = '#E8F000',
    bg_seed_color_global = '#7C00F0',
    contour_color_global = '#0BD9C7',
    fg_seed_color_rgb_global = [232|0, 240|0, 0|0],
    bg_seed_color_rgb_global = [124|0, 0|0, 240|0];

let background_gt_global_context = undefined,
    segmentation_labels_global = undefined,
    background_image_global = undefined,
    background_image_global_context = undefined,
    background_image_global_orig = undefined,
    contour_alpha_global = contour_thickness_vals_global[contour_thickness_vals_global.length - 1];

function getUserInformation() {
    const client = new ClientJS(),
        data = {
            'getSoftwareVersion': client.getSoftwareVersion(),
            //'getBrowserData': client.getBrowserData(),
            'getFingerprint': client.getFingerprint()||false,
            'getUserAgent': client.getUserAgent()||false,
            //'getUserAgentLowerCase': client.getUserAgentLowerCase(),
            'getBrowser': client.getBrowser()||false,
            'getBrowserVersion': client.getBrowserVersion()||false,
            'getBrowserMajorVersion': client.getBrowserMajorVersion()||false,
            'isIE': client.isIE(),
            'isChrome': client.isChrome(),
            'isFirefox': client.isFirefox(),
            'isSafari': client.isSafari(),
            'isMobileSafari': client.isMobileSafari(),
            'isOpera': client.isOpera(),
            'getEngine': client.getEngine(),
            'getEngineVersion': client.getEngineVersion()||false,
            'getOS': client.getOS()||false,
            'getOSVersion': client.getOSVersion()||false,
            'isWindows': client.isWindows(),
            'isMac': client.isMac(),
            'isLinux': client.isLinux(),
            'isUbuntu': client.isUbuntu(),
            'isSolaris': client.isSolaris(),
            'getDevice': client.getDevice()||false,
            'getDeviceType': client.getDeviceType()||false,
            'getDeviceVendor': client.getDeviceVendor()||false,
            'getCPU': client.getCPU()||false,
            'isMobile': client.isMobile(),
            'isMobileMajor': client.isMobileMajor(),
            'isMobileAndroid': client.isMobileAndroid(),
            'isMobileOpera': client.isMobileOpera(),
            'isMobileWindows': client.isMobileWindows(),
            'isMobileBlackBerry': client.isMobileBlackBerry(),
            'isMobileIOS': client.isMobileIOS(),
            'isIphone': client.isIphone(),
            'isIpad': client.isIpad(),
            'isIpod': client.isIpod(),
            //'getScreenPrint': client.getScreenPrint()||false,
            'getColorDepth': client.getColorDepth()||false,
            'getCurrentResolution': client.getCurrentResolution()||false,
            'getAvailableResolution': client.getAvailableResolution()||false,
            'getDeviceXDPI': client.getDeviceXDPI()||false,
            'getDeviceYDPI': client.getDeviceYDPI()||false,
            'getPlugins': client.getPlugins()||false,
            'isJava': client.isJava(),
            'getJavaVersion': client.getJavaVersion()||false,
            'isFlash': client.isFlash(),
            'getFlashVersion': client.getFlashVersion()||false,
            'isSilverlight': client.isSilverlight(),
            'getSilverlightVersion': client.getSilverlightVersion()||false,
            'getMimeTypes': client.getMimeTypes()||false,
            'isMimeTypes': client.isMimeTypes(),
            'isFont': client.isFont(),
            //'getFonts': client.getFonts()||false,
            'isLocalStorage': client.isLocalStorage(),
            'isSessionStorage': client.isSessionStorage(),
            'isCookie': client.isCookie(),
            'getTimeZone': client.getTimeZone()||false,
            'getLanguage': client.getLanguage()||false,
            'getSystemLanguage': client.getSystemLanguage()||false,
            'isCanvas': client.isCanvas(),
            //'getCanvasPrint': client.getCanvasPrint()||false
        };
    return data;
}

function addStartTime() {
    const test_start = (1000 * performance.now()) | 0,
        segmented_data_set_ids = getCookie('segmented_data_set_ids');
    test_result_global[test_start] = {
        'type': 'start',
        'value': test_start,
        'draw_mode': 'foreground',
        'segmentation_method': 'grow_cut',
        'grow_cut_accuracy': Math.pow(2, -25),
        'data_set': data_set_name_global,
        'user_name': user_name_global,
        'user_information': getUserInformation(),
        'all_user_names': getCookie('user_names').split('###')||[],
        'segmented_data_set_ids': (segmented_data_set_ids) ? segmented_data_set_ids.split(',') : []
    };
}

function toISOString(dateItem) {
    // Returns the dateItem information as an ISO standard conform string like "2019-05-15T11:05:18.677Z"
    function pad(number) {
        const r = String(number);
        if (1 === r.length)
            return '0' + r;
        return r;
    }
    return dateItem.getUTCFullYear() + '-' + pad(dateItem.getUTCMonth() + 1)
        + '-' + pad(dateItem.getUTCDate()) + 'T' + pad(dateItem.getUTCHours())
        + ':' + pad(dateItem.getUTCMinutes()) + ':' + pad(dateItem.getUTCSeconds());
}

// Initialize Firebase
(function(){
  const firebaseConfig = {
    apiKey: "AIzaSyDlPmWWLYo8C1CUufeXcgxSlJAk7oLYz8E",
    authDomain: "dummyproject-87e39.firebaseapp.com",
    databaseURL: "https://dummyproject-87e39.firebaseio.com",
    projectId: "dummyproject-87e39",
    storageBucket: "dummyproject-87e39.appspot.com",
    messagingSenderId: "890859699272",
    appId: "1:890859699272:web:e5ebe8ebdcd2ab20"
  };
  firebase.initializeApp(firebaseConfig);
})();

// Get a reference to the database service
const fingerprint = (new ClientJS()).getFingerprint()||'demo_seg',
    overall_ref_global = firebase.database().ref(fingerprint + '/' + toISOString(new Date()));

const onComplete = function(error) {
    // Firebase's check if synchronization was successful
    const msg = (error) ? 'synchronization failed' : 'synchronization succeeded';
    console.log(msg);
    if (!error){
        const segmented_data_set_ids = getCookie('segmented_data_set_ids'),
            current_id = window.location.href.split('#')[1].replace(/,/g, '_');
        if (segmented_data_set_ids){
            const done_segs = new Set(segmented_data_set_ids.split(','));
            done_segs.add(current_id);
            setCookie('segmented_data_set_ids', Array.from(done_segs));
        } else {
            setCookie('segmented_data_set_ids', [current_id]);
        }
        window.location.href = 'index.html#' + current_id.split('@')[0];
    }
};

let number_of_initial_fg_seeds_global = 0,
    number_of_initial_bg_seeds_global = 0;
function initializeSeeds(labels_arr, strengths_arr, width, height){
    const fg_seeds = [], bg_seeds = [];
    // Background seeds on edges
    for (let h = 0; h < height; ++h) {
        const h_times_width = h * width;
        for (let w = 0; w < width; w += (width - 1)) {
            const idx = (h_times_width + w)|0;
            bg_seeds.push([w, h]);
            labels_arr[idx] = -1;
            strengths_arr[idx] = 1;
        }
    }
    for (let w = 1; w < (width - 1); ++w) {
        for (let h = 0; h < height; h += (height -1)) {
            const idx = (h * width + w)|0;
            bg_seeds.push([w, h]);
            labels_arr[idx] = -1;
            strengths_arr[idx] = 1;
        }
    }
    number_of_initial_fg_seeds_global = fg_seeds.length;
    number_of_initial_bg_seeds_global = bg_seeds.length;

    fg_seeds_list_lengths_global = [number_of_initial_fg_seeds_global];
    bg_seeds_list_lengths_global = [number_of_initial_bg_seeds_global];

    return [fg_seeds, bg_seeds];
}

let is_gt_shown_global = false;
function displayGroundTruth(show_gt){
    const c = document.getElementById('canvas'),
        c_p = c.parentNode,
        c_gt = background_gt_global_context.canvas;
    if (show_gt && !is_gt_shown_global){
        if (segmentation_labels_global){
            const context_gt = background_gt_global_context,
                canvas_gt = context_gt.canvas,
                segmentation_labels = segmentation_labels_global;
            context_gt.drawImage(gt_img_global, 0|0, 0|0);

            const image_data = context_gt.getImageData(0|0, 0|0, canvas_gt.width, canvas_gt.height),
                data_buffer = image_data.data,
                len = data_buffer.length,
                fg_alpha = (255 * 0.25)|0,
                bg_alpha = fg_alpha << 1,
                dark = 16|0,
                r_fg = fg_seed_color_rgb_global[0], // 217, 78, 11
                g_fg = fg_seed_color_rgb_global[1],
                b_fg = fg_seed_color_rgb_global[2], // contour_color_rgb_global
                r_bg = bg_seed_color_rgb_global[0],
                g_bg = bg_seed_color_rgb_global[1],
                b_bg = bg_seed_color_rgb_global[2];
            for (let i = 3; i < len; i += 4){
                if (0 > segmentation_labels[(i - 3) >> 2] && (data_buffer[i - 3] || data_buffer[i - 2] || data_buffer[i - 1])) {
                    data_buffer[i - 3] = r_fg;
                    data_buffer[i - 2] = g_fg;
                    data_buffer[i - 1] = b_fg;
                    data_buffer[i] = fg_alpha;
                } else if (0 < segmentation_labels[(i - 3) >> 2] && !(data_buffer[i - 3] || data_buffer[i - 2] || data_buffer[i - 1])){
                    data_buffer[i - 3] = r_bg;
                    data_buffer[i - 2] = g_bg;
                    data_buffer[i - 1] = b_bg;
                    data_buffer[i] = fg_alpha;
                } else {
                    data_buffer[i - 3] = dark;
                    data_buffer[i - 2] = dark;
                    data_buffer[i - 1] = dark;
                    data_buffer[i] = bg_alpha;
                }
            }
            context_gt.putImageData(image_data, 0|0, 0|0);
            test_result_global[(1000 * performance.now())|0] = {
                type: 'show_hint',
                value: true
            };
        }

        c_p.insertBefore(c_gt, c_p.firstChild);
        is_gt_shown_global = true;
    } else {
        // Disable button for a short period of time
        try {
            c_p.removeChild(c_gt);

            const glyphicon = document.getElementById('gt_btn').getElementsByTagName('span')[0];
            glyphicon.parentNode.setAttribute('disabled', true);
            glyphicon.className = 'glyphicon glyphicon-eye-close';
            setTimeout(function(){
                const glyphicon = document.getElementById('gt_btn').getElementsByTagName('span')[0];
                glyphicon.className = 'glyphicon glyphicon-eye-open';
                glyphicon.parentNode.removeAttribute('disabled');
            }, 4000);
            test_result_global[(1000 * performance.now())|0] = {
                type: 'show_hint',
                value: false
            };
        } catch (e) {}
        is_gt_shown_global = false;
    }
}

function setContourThickness(value){
    const new_val = value / 7;
    // Record user action protocol
    test_result_global[(1000 * performance.now())|0] = {
        type: 'transparency_btn',
        value: new_val
    };
    contour_alpha_global = new_val;
    setImageWindowing(window_width_global, window_center_global);
}

function getMedian(array_, is_in_place) {
    if (!array_.length) {
        return 0;
    }
    if (!is_in_place){
        array_ = array_.slice(0);
    }
    const numbers = array_.sort((a, b) => a - b),
        middle = numbers.length >> 1;
    return 0 === numbers.length % 2 ? (numbers[middle] + numbers[middle - 1]) / 2 : numbers[middle];
}

function setAutomaticImageWindowing() {
    if(!background_image_global)  // If used before initialization of image data
        return;

    const tmp_canvas = document.createElement('canvas'),
        tmp_ctx = tmp_canvas.getContext('2d');
    tmp_canvas.width = background_image_global.width;
    tmp_canvas.height = background_image_global.height;

    tmp_ctx.drawImage(background_image_global, 0|0, 0|0);
    const img_buffer = tmp_ctx.getImageData(
        0|0, 0|0, background_image_global.width, background_image_global.height
    ).data;  // pixel_data as Uint8ClampedArray  // copies array

    tmp_ctx.drawImage(gt_img_global, 0|0, 0|0);
    const gt_buffer =  tmp_ctx.getImageData(
        0|0, 0|0, background_image_global.width, background_image_global.height
    ).data;  // pixel_data as Uint8ClampedArray  // copies array

    const len = gt_buffer.length,
        object_values = [];
    for (let i = 0; i < len; i += 4) {
        if (gt_buffer[i]) {
            object_values.push(img_buffer[i]);
        }
    }
    const median_val = getMedian(object_values, true),
        min_val = object_values[0],
        max_val = object_values[object_values.length - 1],
        new_width = Math.max(0, Math.min(255, Math.max(median_val - min_val, max_val - median_val) * 1.1));

    test_result_global[(1000 * performance.now())|0] = {
        type: 'automatic_slider_c',
        value: median_val
    };
    test_result_global[(1000 * performance.now())|0] = {
        type: 'automatic_slider_w',
        value: new_width
    };
    setImageWindowing(new_width, median_val);
}

let window_center_global = 127,
    window_width_global = 255;
function setImageWindowing(new_width, new_center) {
    if(!background_image_global)  // if used before initialization of image data
        return;

    // Save original image if not yet done
    if(!background_image_global_orig){
        const canvas_hidden = document.createElement('canvas'),
            context_hidden = canvas_hidden.getContext('2d');
        canvas_hidden.width = background_image_global.width;
        canvas_hidden.height = background_image_global.height;
        context_hidden.drawImage(background_image_global, 0|0, 0|0);
        background_image_global_orig = context_hidden.getImageData(
            0|0, 0|0, background_image_global.width, background_image_global.height
        ).data;  // pixel_data as Uint8ClampedArray  // copies array
        background_image_global = canvas_hidden;
        background_image_global_context = context_hidden;
    }

    const image_data = background_image_global_context.getImageData(
            0|0, 0|0, background_image_global.width, background_image_global.height
        ),
        data_buffer = image_data.data,
        window_values_min = new_center - (new_width / 2),
        window_values_max = new_center + (new_width / 2),
        normalized_max_window_val = 255.0 / (window_values_max - window_values_min);

    // Apply windowing
    for (let i = 0, end = data_buffer.length; i < end; i += 4){
        const cur_elem = background_image_global_orig[i];
        let new_val = 0;
        if (window_values_min >= cur_elem)
            new_val = 0;
        else if (window_values_max <= cur_elem)
            new_val = 255;
        else {
            new_val = normalized_max_window_val * (cur_elem - window_values_min);
        }
        data_buffer[i] = new_val;
        data_buffer[i + 1] = new_val;
        data_buffer[i + 2] = new_val;
        // Note: [i + 3] is the alpha channel -> leave this one unchanged
    }
    context_global.putImageData(image_data, 0, 0);

    // Draw with updated windowing
    const redraw_background = false;
    drawSegmentationResults(context_global, overlay_context_global, labels_arr_global, redraw_background);

    window_center_global = new_center;
    window_width_global = new_width;
}

function getGTCanvasContextFromBase64(gt_img){
    const canvas_gt = document.createElement('canvas'),
        context_gt = canvas_gt.getContext('2d'),
        canvas = document.getElementById('canvas'),
        new_height = canvas.getBoundingClientRect().height|0,
        new_width = canvas.getBoundingClientRect().width|0;
    canvas_gt.style.position = 'absolute';
    canvas_gt.style.left = canvas.offsetLeft + 'px';
    canvas_gt.style.top = canvas.offsetTop + 'px';
    canvas_gt.height = canvas.height;  // Note: important
    canvas_gt.style.height = new_height;
    canvas_gt.width = canvas.style.width;  // Note: important
    canvas_gt.style.width = canvas.style.width;  // Note: important
    canvas_gt.style.left = canvas.offsetLeft + 'px';
    canvas_gt.style.top = canvas.offsetTop + 'px';
    canvas_gt.width = canvas.width;  // Note: important
    canvas_gt.style.width = new_width + 'px';  // Note: important

    canvas_gt.addEventListener('contextmenu', function(){}, false);
    context_gt.imageSmoothingEnabled = false;
    context_gt.msImageSmoothingEnabled = false;
    context_gt.imageSmoothingEnabled = false;
    context_gt.drawImage(gt_img, 0|0, 0|0);

    const image_data = context_gt.getImageData(0|0, 0|0, canvas_gt.width, canvas_gt.height),
        data_buffer = image_data.data,
        len = data_buffer.length,
        fg_alpha = (255 * 0.3)|0,
        bg_alpha = fg_alpha, // << 1,
        dark = 20|0,
        r = 217, //fg_seed_color_rgb_global[0],  // 217, 78, 11
        g = 78, //fg_seed_color_rgb_global[1],
        b = 11; //fg_seed_color_rgb_global[2]; // contour_color_rgb_global
    for (let i = 3; i < len; i += 4){
        if (data_buffer[i - 3] || data_buffer[i - 2] || data_buffer[i - 1]){
            data_buffer[i - 3] = r;
            data_buffer[i - 2] = g;
            data_buffer[i - 1] = b;
            data_buffer[i] = fg_alpha;
        } else {
            data_buffer[i - 3] = dark;
            data_buffer[i - 2] = dark;
            data_buffer[i - 1] = dark;
            data_buffer[i] = bg_alpha;
        }
    }
    context_gt.putImageData(image_data, 0|0, 0|0);

    return context_gt;
}

function getDataFromImage(image){
    const canvas = document.getElementById('canvas'),
        context = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    context_global = context;
    image_global = image;
    context.imageSmoothingEnabled = false;
    context.msImageSmoothingEnabled = false;
    context.imageSmoothingEnabled = false;
    context.drawImage(image, 0|0, 0|0);
    image_data = context.getImageData(0|0, 0|0, image.width, image.height);  // is a copy
    pixel_data_rgba = image_data.data;

    const old_height = parseInt(canvas.style.height.slice(0, -2));
    canvas.style.height = '10px';
    const new_height = (Math.max(
            document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight,
            document.documentElement.scrollHeight, document.documentElement.offsetHeight
        ) - canvas.getBoundingClientRect().top - 20)|0;
    canvas.style.height = new_height + 'px';
    canvas.style.width = ((new_height / old_height) * parseInt(canvas.style.width.slice(0, -2))) + 'px';

    canvas.imageSmoothingEnabled = false;
    canvas.msImageSmoothingEnabled = false;
    canvas.imageSmoothingEnabled = false;

    const overlay_canvas = document.createElement('canvas');
    overlay_canvas.style.position = 'absolute';
    overlay_canvas.style.left = canvas.offsetLeft + 'px';
    overlay_canvas.style.top = canvas.offsetTop + 'px';
    overlay_context_global = overlay_canvas.getContext('2d');
    overlay_canvas.height = new_height;  // Note: important
    overlay_canvas.style.height = new_height;
    overlay_canvas.width = canvas.style.width;  // Note: important
    overlay_canvas.style.width = canvas.style.width;  // Note: important
    overlay_canvas.style.cursor = 'crosshair';
    canvas.parentNode.insertBefore(overlay_canvas, canvas);

    overlay_canvas.style.left = canvas.offsetLeft + 'px';
    overlay_canvas.style.top = canvas.offsetTop + 'px';
    const new_width = canvas.getBoundingClientRect().width|0;
    overlay_canvas.width = new_width;  // Note: important
    overlay_canvas.style.width = new_width + 'px';  // Note: important

    overlay_canvas.addEventListener('contextmenu', function(){}, false);
    overlay_canvas.addEventListener('mousedown', mouseDown, false);
    overlay_canvas.addEventListener('mouseup', mouseUp, false);
    overlay_canvas.addEventListener('mousemove', addSeedPoint, false);

    overlay_canvas.addEventListener('touchstart', function(e){
        const touch_evnt = e.changedTouches[0]; // Reference first touch point (ie: first finger)
        left_mouse_button_is_pressed_global = true;
        right_mouse_button_is_pressed_global = false;
        addSeedPoint(touch_evnt);
        e.preventDefault();
    }, false);

    overlay_canvas.addEventListener('touchmove', function(e){
        const touch_evnt = e.changedTouches[0]; // Reference first touch point for this event
        addSeedPoint(touch_evnt);
        e.preventDefault();
    }, false);

    overlay_canvas.addEventListener('touchend', function(e){
        left_mouse_button_is_pressed_global = false;
        right_mouse_button_is_pressed_global = false;
        logEndOfAction();
        segmentAndDraw();
        e.preventDefault();
    }, false);

    const pixel_data = pixel_data_global = new Uint8Array(pixel_data_rgba.length >> 2);
    width_global = image_data.width;
    height_global = image_data.height;
    labels_arr_global = new Int8Array(pixel_data_rgba.length >> 2);
    strengths_arr_global = new Float32Array(pixel_data_rgba.length >> 2);

    scale_factor_global = height_global / new_height;

    // Populate image array
    let l = (pixel_data.length - (pixel_data.length % 4))|0;  // Note: or ((pixel_data.length >> 2) << 2)
    while(l--) {
        pixel_data[l] = pixel_data_rgba[l * 4];
    }

    // Get min and max val
    let ll = pixel_data.length, min_val = 999|0, max_val = 0|0;
    while(ll--) {
        if (min_val > image[ll]){
            min_val = image[ll];
        }
        if (max_val < image[ll]){
            max_val = image[ll];
        }
    }
    min_val_global = min_val;
    max_val_global = max_val;

    const seeding_res = initializeSeeds(labels_arr_global, strengths_arr_global, width_global, height_global);
    // Extend an existing array fg_seeds_list_global with another array
    // same as "fg_seeds_list_global.push.apply(fg_seeds_list_global, seeding_res[0])" or
    // "fg_seeds_list_global.push(...seeding_res[0])"
    Array.prototype.push.apply(fg_seeds_list_global, seeding_res[0]);
    Array.prototype.push.apply(bg_seeds_list_global, seeding_res[1]);

    window.requestAnimationFrame(drawMain);
}

let next_timestamp_global = 0|0,
    next_seed_list_length_fg_global = 0,
    next_seed_list_length_bg_global = 0;
function drawMain(timestamp){
    if(timestamp > next_timestamp_global) {
        next_timestamp_global = timestamp + 1;  // Timestamp is in seconds
        if (
            !left_mouse_button_is_pressed_global && !right_mouse_button_is_pressed_global &&
            (
                (next_seed_list_length_fg_global < fg_seeds_list_global.length) ||
                (next_seed_list_length_bg_global < bg_seeds_list_global.length)
            )
        ){
            next_seed_list_length_fg_global = fg_seeds_list_global.length;
            next_seed_list_length_bg_global = bg_seeds_list_global.length;
            segmentAndDraw();

            // Note: quick fix!
            setImageWindowing(window_width_global, window_center_global);
        }
    }
    window.requestAnimationFrame(drawMain);
}

let last_point_added_global = [0, 0];
function addSeedPoint(evt){
    if(!left_mouse_button_is_pressed_global && !right_mouse_button_is_pressed_global)
        return;
    // Rect is a DOMRect object with four properties: left, top, right, bottom
    const rect = evt.target.getBoundingClientRect(),
        seed_list = left_mouse_button_is_pressed_global ? fg_seeds_list_global : bg_seeds_list_global,
        last_seed = (seed_list.length) ? seed_list[seed_list.length - 1] : [-1, -1],
        new_point = [
            ((evt.clientX - rect.left) * scale_factor_global)|0,  // w
            ((evt.clientY - rect.top) * scale_factor_global)|0   // h
        ];
    // Ignore if new point is same point as last time (relevant when drawing lines and shapes)
    if (new_point[0] === last_seed[0] && new_point[1] === last_seed[1])
        return;
    //
    seed_list.push(new_point);
    last_point_added_global = new_point;

    const idx = new_point[1] * width_global + new_point[0];
    labels_arr_global[idx] = left_mouse_button_is_pressed_global ? 1|0 : -1|0;
    strengths_arr_global[idx] = 1;
    overlay_context_global.fillStyle = overlay_context_global.strokeStyle =
        left_mouse_button_is_pressed_global ? fg_seed_color_global : bg_seed_color_global;
    overlay_context_global.fillRect(new_point[0], new_point[1], 1, 1);

    recordInteraction(12);  // log: add new seed point
}

function segmentAndDraw() {
    const start_time = (1000 * performance.now())|0;
    const pixel_data = pixel_data_global,
        labels_arr = labels_arr_global,
        strengths_arr = strengths_arr_global,
        width = width_global,
        height = height_global,
        min_val = min_val_global,
        max_val = max_val_global,
        accuracy = Math.pow(2, -25), // 2**-25 == 2.98023e-8, using a power of two for faster computation
        res = growCut(pixel_data, labels_arr, strengths_arr, height, width, min_val, max_val, accuracy),
        new_labels = res[0], // open_binary(res[0], height, width, opening_iterations);
        end_time = (1000 * performance.now())|0;

    segmentation_labels_global = new_labels;

    test_result_global[end_time] = {
        type: 'time_spent_segmenting',
        value: (end_time - start_time)
    };

    const redraw_background = false;
    drawSegmentationResults(context_global, overlay_context_global, new_labels, redraw_background);
}

function drawSegmentationResults(img_context, context, new_labels, redraw_background){
    if (undefined === redraw_background || redraw_background){
        img_context.clearRect(0|0, 0|0, img_context.canvas.width, img_context.canvas.height);
        img_context.drawImage(image_global, 0|0, 0|0);
    }

    if (!context){
        return;
    }

    context.clearRect(0|0, 0|0, context.canvas.width, context.canvas.height);
    // Context is the overlay context
    const width = (width_global|0),
        height = (height_global|0),
        overlay_scale = (context.canvas.getBoundingClientRect().width / img_context.canvas.width),
        // Note: is ca. 1 pixel size but without interpolation, therefore more crisp
        contour_thickness = (overlay_scale >> 0) / overlay_scale,
        contour_thickness_offset = 0;

    context.setTransform(overlay_scale, 0, 0, overlay_scale, 0, 0);
    context.fillStyle = context.strokeStyle = contour_color_global;
    context.globalAlpha = 1|0;
    let h = height;
    while(h--){
        const h_times_width = h * width;
        let w = width;
        while(w--){
            const idx = h_times_width + w;
            if (1 === new_labels[idx]){
                // Check if on contour line
                if(!(1 === new_labels[idx - width - 1] && 1 === new_labels[idx - width] && 1 === new_labels[idx - width + 1] &&
                    1 === new_labels[idx - 1] && 1 === new_labels[idx + 1] &&
                    1 === new_labels[idx + width - 1] && 1 === new_labels[idx + width] && 1 === new_labels[idx + width + 1])
                ){
                    context.fillRect(w - contour_thickness_offset, h - contour_thickness_offset, contour_thickness, contour_thickness);
                }
            }
        }
    }

    context.fillStyle = context.strokeStyle = fg_seed_color_global;
    const fg_seeds = fg_seeds_list_global;
    let idx_fg = fg_seeds_list_global.length;
    while(idx_fg--) {
        context.fillRect(fg_seeds[idx_fg][0] - contour_thickness_offset,
            fg_seeds[idx_fg][1] - contour_thickness_offset,
            contour_thickness, contour_thickness);
    }
    context.fillStyle = context.strokeStyle = bg_seed_color_global;
    const bg_seeds = bg_seeds_list_global;
    let idx_bg = bg_seeds_list_global.length;
    while (idx_bg-- > number_of_initial_bg_seeds_global) {
        context.fillRect(bg_seeds[idx_bg][0] - contour_thickness_offset,
            bg_seeds[idx_bg][1] - contour_thickness_offset,
            contour_thickness, contour_thickness);
    }

    // Note: contour line and seeds/scribbles do have the same opacity after this operation
    const current_alpha = (255 * Math.min(1, contour_alpha_global))|0,
        current_alpha_reduced = (0.85 * 255 * Math.min(1, contour_alpha_global))|0,
        fg_color_r = fg_seed_color_rgb_global[0],
        bg_color_r = bg_seed_color_rgb_global[0];
    //if (255 !== current_alpha){
    // Alter alpha value of contour line (workaround to compensate for overlapping rectangles)
    const image_data = context.getImageData(0|0, 0|0, context.canvas.width, context.canvas.height),
        data_buffer = image_data.data,
        len = data_buffer.length;
    for (let i = 3; i < len; i += 4){
        if (data_buffer[i - 3] || data_buffer[i - 2] || data_buffer[i - 1]){
            if (fg_color_r === data_buffer[i - 3] ||
                bg_color_r === data_buffer[i - 3]) {
                data_buffer[i] = current_alpha_reduced;
            } else {
                data_buffer[i] = current_alpha;
            }
        }
    }
    context.putImageData(image_data, 0|0, 0|0);
    //}

    context.globalAlpha = 1|0;
}

let label_next_array_global = false,
    strengths_next_arr_global = false;
function growCut(image, labels_arr, strengths_arr, dim0_u, dim1_u, min_val, max_val, accuracy){
    if (!label_next_array_global){
        label_next_array_global = (1 === labels_arr.BYTES_PER_ELEMENT) ? (new Int8Array(labels_arr.length)) : (new Int16Array(labels_arr.length));
    }
    if (!strengths_next_arr_global){
        strengths_next_arr_global = (4 === strengths_arr.BYTES_PER_ELEMENT) ? (new Float32Array(strengths_arr.length)) : (new Float64Array(strengths_arr.length));
    }

    const ws_u = 1|0,
        accur = (!!accuracy) ? accuracy : 5e-8;
    let labels_next_arr = label_next_array_global,
        strengths_next_arr = strengths_next_arr_global,
        changes = true,
        iters = 1000,
        dd0_min, dd0_max,
        dd1_min, dd1_max;

    // Get min and max val if not present
    if (undefined === min_val || undefined === max_val){
        let l = image.length, min_val_ = 255|0, max_val_ = 0|0;
        while (l--) {
            if (min_val_ > image[l]){
                min_val_ = image[l];
            }
            if (max_val_ < image[l]){
                max_val_ = image[l];
            }
        }
        min_val = min_val_;
        max_val = max_val_;
    }
    const one_div_by_max_distance_squared = (1.0 / (max_val - min_val)) * (1.0 / (max_val - min_val));

    while(changes && iters--){
        changes = false;

        let d0_u = dim0_u;
        while (d0_u--){
            if (ws_u >= d0_u) {
                dd0_min = 0;
                dd0_max = d0_u + ws_u + 1;
            } else {
                dd0_min = d0_u - ws_u;
                if (dim0_u > (d0_u + ws_u)) {  // if equal after +1, take either
                    dd0_max = d0_u + ws_u + 1;
                } else {
                    dd0_max = dim0_u;
                }
            }
            //
            let d1_u = dim1_u;
            while (d1_u--){
                //
                if (ws_u >= d1_u) {
                    dd1_min = 0;
                    dd1_max = d1_u + ws_u + 1;
                } else {
                    dd1_min = d1_u - ws_u;
                    if (dim1_u > (d1_u + ws_u)) {  // if equal after +1, take either
                        dd1_max = d1_u + ws_u + 1;
                    } else {
                        dd1_max = dim1_u;
                    }
                }

                //
                const outer_idx = (d0_u * dim1_u + d1_u),
                    defense_strength_orig = (strengths_arr[outer_idx] + accur),
                    current_cell_value = image[outer_idx],
                    dd0_max_times_dim1_u = (dd0_max * dim1_u);
                let defense_strength = strengths_arr[outer_idx],
                    winning_colony = labels_arr[outer_idx];

                for (let dd0_u_times_dim1_u = dd0_min * dim1_u; dd0_u_times_dim1_u < dd0_max_times_dim1_u; dd0_u_times_dim1_u += dim1_u) {
                    const dd0_u_times_dim1_u_plus_dd1_max = dd0_u_times_dim1_u + dd1_max;
                    for (let inner_idx = dd0_u_times_dim1_u + dd1_min; inner_idx < dd0_u_times_dim1_u_plus_dd1_max; ++inner_idx) {
                        // Inexpensive test to reduce number of computations: if cell cannot be conquered by
                        // the attacker, do not investigate further and continue with current cell's next neighbor.
                        if (defense_strength_orig >= strengths_arr[inner_idx])
                            continue;

                        // p -> current cell, (outer_idx, d2_u)
                        // q -> attacker, (dd0, dd1, dd2)
                        // attack_strength = g(distance(image, outer_idx, inner_idx)
                        const dist = (current_cell_value - image[inner_idx]);
                        const attack_strength = (1.0 - (one_div_by_max_distance_squared * (dist * dist))) *
                            strengths_arr[inner_idx];

                        if (attack_strength > defense_strength_orig) {
                            // Differentiate here to increase cell changes independent of the neighbors' ordering
                            if (attack_strength > (defense_strength + accur)) {
                                defense_strength = attack_strength;
                                winning_colony = labels_arr[inner_idx];
                                changes = true;
                            }
                        }
                    }
                }
                labels_next_arr[outer_idx] = winning_colony;
                strengths_next_arr[outer_idx] = defense_strength;
            }
        }

        const tmp = labels_arr;
        labels_arr = labels_next_arr;
        labels_next_arr = tmp;

        const tmp2 = strengths_arr;
        strengths_arr = strengths_next_arr;
        strengths_next_arr = tmp2;

    }  // END while
    return [labels_next_arr, strengths_next_arr];
}

let left_mouse_button_is_pressed_global = false,
    right_mouse_button_is_pressed_global = false,
    set_mode_val_global = 1; // If user clicks on image, they draw fg-seeds
function mouseDown(evt){
    if(1 === evt.which){
        left_mouse_button_is_pressed_global = (1 === set_mode_val_global);
        right_mouse_button_is_pressed_global = (1 !== set_mode_val_global);
    } else {
        left_mouse_button_is_pressed_global = (1 !== set_mode_val_global);
        right_mouse_button_is_pressed_global = (1 === set_mode_val_global);
    }
    addSeedPoint(evt);
}

function mouseUp(){
    left_mouse_button_is_pressed_global = false;
    right_mouse_button_is_pressed_global = false;
    logEndOfAction();
    segmentAndDraw();
}

// Choose mode for drawing (change seed label to draw with same mouseclick)
function setMode(val){
    const mouse_icon = document.getElementById('mouselabelhint');
    if (0 === val){ // background drawing mode
        set_mode_val_global = 0;
        mouse_icon.src = 'mouse_bf.svg';
    } else if (1 === val){ // foreground drawing mode
        set_mode_val_global = 1;
        mouse_icon.src = 'mouse_fb.svg';
    }
}

// Record a user's actions
function recordInteraction(val_action){
    //  0: decision 0 (only in guided version)
    //  1: decision 1 (only in guided version)
    //  2: decision 2 (only in guided version)
    //  3: decision 3 (only in guided version)
    //  4: back button (is recorded in function undoLastAction())
    //  5: confirm decision
    //  6: slider center (is recorded in window.addEventListener('windowing_slider_event')...
    //  7: slider width (is recorded in window.addEventListener('windowing_slider_event')...
    //  8: transparency btn (is recorded in function next_contour_thickness())
    //  9: mark background
    // 10: mark object
    // 11: help button (not used)
    // 12: new seed point (foreground = 1, background = 0)

    if (5 === val_action){ // confirm decision
        test_result_global[(1000 * performance.now())|0] = {
            type: 'confirm_btn'
        };
    }
    else if (9 === val_action){ // mark background
        test_result_global[(1000 * performance.now())|0] = {
            type: 'mark_bg_btn'
        };
    }
    else if (10 === val_action){ // mark object
        test_result_global[(1000 * performance.now())|0] = {
            type: 'mark_fg_btn'
        };
    }
    else if (12 === val_action) { // new seed point
        if (1 === set_mode_val_global) {
            test_result_global[(1000 * performance.now())|0] = {
                type: 'seed',
                w: last_point_added_global[0],
                h: last_point_added_global[1],
                label: left_mouse_button_is_pressed_global ? 1 : -1
            };
        }
        else if (0 === set_mode_val_global) {
            test_result_global[(1000 * performance.now())|0] = {
                type: 'seed',
                w: last_point_added_global[0],
                h: last_point_added_global[1],
                label: left_mouse_button_is_pressed_global ? -1 : 1
            }
        }
    }
}

// Note: during startup, those arrays are overwritten by the initializeSeeds function!
let fg_seeds_list_lengths_global = [0],
    bg_seeds_list_lengths_global = [0];
function logEndOfAction(){
    fg_seeds_list_global = uniqueBy(fg_seeds_list_global);
    bg_seeds_list_global = uniqueBy(bg_seeds_list_global);
    fg_seeds_list_lengths_global.push(fg_seeds_list_global.length);
    bg_seeds_list_lengths_global.push(bg_seeds_list_global.length);
}

function uniqueBy(a, key) {
    if (!key){
        key = JSON.stringify;
    }
    const seen = {};
    return a.filter(function(item) {
        const k = key(item);
        return seen.hasOwnProperty(k) ? false : (seen[k] = true);
    })
}

function undoLastAction(){
    if (1 === fg_seeds_list_lengths_global.length){
        return;
    }
    const fg_list_length_after_action = fg_seeds_list_lengths_global.pop(),
        bg_list_length_after_action = bg_seeds_list_lengths_global.pop(),
        fg_list_length_before_action = fg_seeds_list_lengths_global[fg_seeds_list_lengths_global.length - 1],
        bg_list_length_before_action = bg_seeds_list_lengths_global[bg_seeds_list_lengths_global.length - 1];

    let fg_seeds_deleted = [],
        bg_seeds_deleted = [];
    const fg_len_diff = fg_list_length_after_action - fg_list_length_before_action,
        bg_len_diff = bg_list_length_after_action - bg_list_length_before_action;
    if (0 !== fg_len_diff){
        fg_seeds_deleted = fg_seeds_list_global.splice(-fg_len_diff);
    }
    // Note: this should be an "else if" or "else" as well. Is an "if"-only,
    // just to be extra sure to remain compatible with future versions.
    if (0 !== bg_len_diff){
        bg_seeds_deleted = bg_seeds_list_global.splice(-bg_len_diff);
    }

    let fg_idx = fg_seeds_list_global.length,
        bg_idx = bg_seeds_list_global.length;
    const image_width = width_global,
        labels_arr = labels_arr_global,
        strength_arr = strengths_arr_global;
    labels_arr.fill(0);
    strength_arr.fill(0.0);
    while(fg_idx--){
        const current_point = fg_seeds_list_global[fg_idx],  // w, h
            current_idx = current_point[1] * image_width + current_point[0];
        labels_arr[current_idx] = 1;
        strength_arr[current_idx] = 1.0;
    }
    while(bg_idx--){
        const current_point = bg_seeds_list_global[bg_idx],  // w, h
            current_idx = current_point[1] * image_width + current_point[0];
        labels_arr[current_idx] = -1;
        strength_arr[current_idx] = 1.0;
    }

    // Trigger a recalculation via grow cut segmentation
    next_seed_list_length_fg_global = 0;
    next_seed_list_length_bg_global = 0;

    // Add to protocol for user actions
    test_result_global[(1000 * performance.now())|0] = {
        type: 'undo_btn',
        fg_seeds_deleted: fg_seeds_deleted,
        bg_seeds_deleted: bg_seeds_deleted
    };
}

let has_confirmed_before_global = false;
function confirmDecision() {
    if (has_confirmed_before_global){
        return;
    } else {
        has_confirmed_before_global = true;
    }
    recordInteraction(5);  // confirm result
    // send test result to the firebase server
    if (!!overall_ref_global){
        // same as the previous example, except we will also log a message, when the data has finished synchronizing
        overall_ref_global.set(test_result_global, onComplete);
    }
}

// Start interaction ---------------------------------

let image_data = undefined,  // ImageData
    pixel_data_rgba = undefined,  // Uint8ClampedArray
    image_global = null,
    fg_seeds_list_global = [],
    bg_seeds_list_global = [],
    pixel_data_global = null,
    labels_arr_global = null,
    strengths_arr_global = null,
    width_global = null,
    height_global = null,
    min_val_global = 999|0,
    max_val_global = 0|0,
    context_global = null,
    overlay_context_global = null,
    scale_factor_global = 1,
    ni = new Image(),
    gt_img_global = new Image();

ni.onload = function () {getDataFromImage(ni); background_image_global = ni;};
ni.src = overall_image_data_global['img'];  // base64 encoded string

setTimeout(function(){
    gt_img_global.onload = function(){background_gt_global_context = getGTCanvasContextFromBase64(gt_img_global);};
    gt_img_global.src = overall_image_data_global['gt'];  // base64 encoded string
}, 800);

setTimeout(function(){
    document.getElementById('confirm_btn').removeAttribute('disabled');
}, 4000);

window.addEventListener('windowing_slider_event', function(evnt){
    // console.log(evnt.newValue);
    const tmp = evnt.newValue.split(','),
        window_width = parseInt(tmp[0]),
        window_center = parseInt(tmp[1]);

    //  6: slider center
    test_result_global[(1000 * performance.now())|0] = {
        type: 'slider_c',
        value: window_center
    };

    //  7: slider width
    test_result_global[(1000 * performance.now())|0] = {
        type: 'slider_w',
        value: window_width
    };

    setImageWindowing(window_width, window_center);
});

window.addEventListener('contour_slider_event', function(evnt){
    setContourThickness(parseFloat(evnt.newValue));
});

document.getElementById('back_btn').addEventListener('mouseup', function () {undoLastAction();}, false);
document.getElementById('confirm_btn').addEventListener('mouseup', function () {confirmDecision();}, false);
document.getElementById('add_foreground').addEventListener('mouseup', function () {setMode(1);}, false);
document.getElementById('add_background').addEventListener('mouseup', function () {setMode(0);}, false);
//document.getElementById('myContourThickness').addEventListener('click', function () {next_contour_thickness();}, false);
document.getElementById('gt_btn').addEventListener('mouseup', function () {displayGroundTruth(true);}, false);
//document.getElementById('gt_btn').addEventListener('mouseup', function () {displayGroundTruth(false);}, false);
document.getElementById('canvas').parentNode.addEventListener('mouseenter', function () {displayGroundTruth(false);}, false);
document.getElementById('auto_window').parentNode.addEventListener('mouseup', setAutomaticImageWindowing, false);

document.getElementById('restart_btn').addEventListener('mouseup', function() {window.location.reload();}, false);

addStartTime();
