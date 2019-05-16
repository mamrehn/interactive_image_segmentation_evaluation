(function() {
    let name = false,
        choose = false,
        parent = false,
        div = false;

    function set_cookie(key, value) {
        document.cookie = key + '=' + value + '; path=/';
    }

    function get_cookie(key) {
        const name_eq = key + '=',
            cookie_parts = document.cookie.split(';'),
            len = cookie_parts.length;
        for (let i = 0; i < len; ++i) {
            let current_info = cookie_parts[i].trim();
            if (0 === current_info.indexOf(name_eq))
                return current_info.substring(name_eq.length, current_info.length);
        }
        return null;
    }

    function jump_to_first_item_in_randomized_list(data_sets, user_name) {
        const segmented_data_set_ids = get_cookie('segmented_data_set_ids') || [];
        for (let link_name of data_sets) {
            if (!segmented_data_set_ids.includes(user_name + '@' + link_name)) {
                window.location.href = 'segment.html#' + user_name + '@' + link_name;
                break;
            }
        }
    }

    function edit_urls() {
        const text = document.getElementById('text');
        if (text) {
            name = text.value.trim();
            if (name && 3 < name.length) {
                window.location.href += '#' + name;
                parent.removeChild(div);
                parent.appendChild(choose);
            }
        }
        if (name && 3 < name.length) {
            const data_sets = get_overall_image_data_global(null);

            // Update cookie
            const user_names = get_cookie('user_names');
            if (null === user_names) {
                set_cookie('user_names', name);
            } else {
                const un_d = {};
                un_d[name] = true;
                for (let u of user_names.split('###')) {
                    un_d[u] = true;
                }
                set_cookie('user_names', Object.keys(un_d).join('###'));
            }
            //

            const data_sets_randomized = randomize(data_sets, name);
            jump_to_first_item_in_randomized_list(data_sets_randomized, name);
            return;
        } else {
            text.value = text.value.trim();
            alert('Please use a name with 4 or more characters.')
        }
    }

    function randomize(items, seed_str) {
        let temp, idx = items.length;
        // Sets Math.random to a PRNG initialized using the given explicit seed.
        Math.seedrandom(seed_str);
        while (--idx) {
            const new_index = (idx * Math.random()) | 0;
            temp = items[new_index];
            items[new_index] = items[idx];
            items[idx] = temp;
        }
        return items;
    }

    if (1 < window.location.href.split('@').length) {
        window.location.href = window.location.href.split('@')[0];
    }

    const url = window.location.href.split('#');
    name = (1 > url.length) ? false : url[1];
    name = (!name || 1 > name.length) ? false : name.split('@')[0].replace(/,/g, '_');
    if (!!name) {
        edit_urls();
    } else {
        choose = document.getElementById('choose');
        parent = choose.parentNode;
        parent.removeChild(choose);

        div = document.createElement('div');
        div.className = 'center';
        const form = document.createElement('input');
        form.type = 'text';
        form.id = 'text';
        form.placeholder = 'Your name or pseudonym here..';
        form.style.width = '70%';
        form.style.fontSize = '40%';
        const confirm = document.createElement('button');
        confirm.innerHTML = 'Start tests';
        confirm.style.width = '25%';
        confirm.style.fontSize = '40%';
        confirm.addEventListener('click', edit_urls, false);

        div.appendChild(form);
        div.appendChild(confirm);
        parent.appendChild(div);

        const user_names_str = get_cookie('user_names');
        if (null !== user_names_str)
            $(form).autocomplete({source: user_names_str.split('###')});
    }
})();
