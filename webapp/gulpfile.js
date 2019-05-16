
const gulp = require('gulp'),
    inline = require('gulp-inline'),
    uglify = require('gulp-uglify'),
    htmlmin = require('gulp-htmlmin'),
    cssmin = require('gulp-minify-css'),
    concat = require('gulp-concat'),
    pump = require('pump'),
    babel = require('gulp-babel'),
    inlinemin = require('gulp-minify-inline');

const paths = {
    scripts_seg: ['script/demo_manual.js'],  // ['!script/**/*.min.js', 'script/**/*.js'],
    scripts_idx: ['script/index.js'],  // ['!script/**/*.min.js', 'script/**/*.js'],
    scripts_data: ['script/img_data.js'],
    html_seg: ['segment.html'],
    html_idx: ['index.html'],
    css: ['css/custom.css']  // ['script/**/*.min.css', 'css/simple-sidebar.css']
};

/*
gulp.task('minifyjs', function() {
    gulp.src(paths.scripts)
        .pipe(concat('all.js'))
        .pipe(uglify())
        .pipe(gulp.dest('./deploy/'))
});
*/

/*
gulp.task('minifyjs', function (cb) {
    pump([
            gulp.src(paths.scripts),
            concat('all.js'),
            uglify(),
            gulp.dest('./deploy/')
        ],
        cb
    );
});
*/

// http://stackoverflow.com/a/39169660
gulp.task('minifyjs_seg', (cb) => {
    pump([
        gulp.src(paths.scripts_seg),
        concat('demo_manual.js'),
        babel({
            presets: ['babili'],
            comments: false
        }),
        gulp.dest('./deploy/script/')
        ],
        cb
    );
});

gulp.task('minifyjs_idx', (cb) => {
    pump([
            gulp.src(paths.scripts_idx),
            concat('index.js'),
            babel({
                presets: ['babili'],
                comments: false
            }),
            gulp.dest('./deploy/script/')
        ],
        cb
    );
});

gulp.task('minifyjs_data', (cb) => {
    pump([
            gulp.src(paths.scripts_data),
            concat('img_data.js'),
            babel({
                presets: ['babili'],
                comments: false
            }),
            gulp.dest('./deploy/script/')
        ],
        cb
    );
});

gulp.task('minifyhtml_idx', (cb) => {
    pump([
        gulp.src(paths.html_idx),
        concat('index.html'),
        htmlmin({
            collapseWhitespace: true,
            removeComments: true
        }),
        inlinemin(),
        gulp.dest('./deploy/')
        ],
        cb
    );
});

gulp.task('minifyhtml_seg', (cb) => {
    pump([
            gulp.src(paths.html_seg),
            concat('segment.html'),
            htmlmin({
                collapseWhitespace: true,
                removeComments: true
            }),
            inlinemin(),
            gulp.dest('./deploy/')
        ],
        cb
    );
});

gulp.task('minifcss', (cb) => {
    pump([
        gulp.src(paths.css),
        concat('custom.css'),
        cssmin({
            collapseWhitespace: true,
            removeComments: true
        }),
        gulp.dest('./deploy/css/')
        ],
        cb
    );
});

gulp.task('mininline_seg', (cb) => {
    pump([
        gulp.src(paths.html_seg),
        htmlmin({
            collapseWhitespace: true,
            removeComments: true
        }),
        inlinemin(),
        inline({
            base: './',
            js: () => babel({
                presets: ['babili'],
                comments: false,
                // https://github.com/babel/babili/issues/54
                /*mangle: {
                    keepFnName: false
                },
                deadcode: true,
                flipComparisons: true,
                removeConsole: true*/
            }),
            css: () => cssmin({
                collapseWhitespace: true,
                removeComments: true
            }),  // [cssmin, autoprefixer({ browsers:['last 2 versions'] })],
            // disabledTypes: ['svg', 'img'], // Only inline css files
            // ignore: ['./css/do-not-inline-me.css']
        }),
        concat('segment.html'),
        gulp.dest('./deploy/')
        ],
        cb
    );
});

gulp.task('mininline_idx', (cb) => {
    pump([
            gulp.src(paths.html_idx),
            htmlmin({
                collapseWhitespace: true,
                removeComments: true
            }),
            inlinemin(),
            inline({
                base: './',
                js: () => babel({
                    presets: ['babili'],
                    comments: false,
                    // https://github.com/babel/babili/issues/54
                    /*mangle: {
                     keepFnName: false
                     },
                     deadcode: true,
                     flipComparisons: true,
                     removeConsole: true*/
                }),
                css: () => cssmin({
                    collapseWhitespace: true,
                    removeComments: true
                }),  // [cssmin, autoprefixer({ browsers:['last 2 versions'] })],
                // disabledTypes: ['svg', 'img'], // Only inline css files
                // ignore: ['./css/do-not-inline-me.css']
            }),
            concat('index.html'),
            gulp.dest('./deploy/')
        ],
        cb
    );
});

// ['minifyjs', 'minifyhtml', 'minifcss']); // 'mininline_seg', 'mininline_idx',
gulp.task('default', ['minifyhtml_seg', 'minifyhtml_idx', 'minifyjs_seg', 'minifyjs_idx', 'minifyjs_data', 'minifcss']);

// configure which files to watch and what tasks to use on file changes
/*gulp.task('watch', function() {
    gulp.watch(paths.scripts, ['minifyjs', 'minifyhtml']);
});
*/
