module.exports = function(grunt) {
  // load all grunt tasks
  require('matchdep').filterDev('grunt-*').forEach(grunt.loadNpmTasks);

  var envJSON = grunt.file.readJSON(".env.json");
  var PROJECT_DIR = "docs/";

  switch (grunt.option('project')) {
    case "docs":
      PROJECT_DIR = envJSON.DOCS_DIR;
      break;
    case "tutorials":
      PROJECT_DIR = envJSON.TUTORIALS_DIR;
      break;
   }

  grunt.initConfig({
    // Read package.json
    pkg: grunt.file.readJSON("package.json"),

    open : {
      dev: {
        path: 'http://localhost:1919'
      }
    },

    connect: {
      server: {
        options: {
          port: 1919,
          base: 'docs/build',
          livereload: true
        }
      }
    },
    copy: {
      fonts: {
        files: [
          {
              expand: true,
              flatten: true,
              src: ['fonts/FreightSans/*'],
              dest: 'pytorch_sphinx_theme/static/fonts/FreightSans',
              filter: 'isFile'
          },

          {
              expand: true,
              flatten: true,
              src: ['fonts/IBMPlexMono/*'],
              dest: 'pytorch_sphinx_theme/static/fonts/IBMPlexMono',
              filter: 'isFile'
          }
        ]
      },

      images: {
        files: [
          {
              expand: true,
              flatten: true,
              src: ['images/*'],
              dest: 'pytorch_sphinx_theme/static/images',
              filter: 'isFile'
          }
        ]
      },

      vendor: {
        files: [
          {
              expand: true,
              cwd: 'node_modules/bootstrap/scss/',
              src: "**/*",
              dest: 'scss/vendor/bootstrap',
              filter: 'isFile'
          },

          {
            expand: true,
            flatten: true,
            src: [
              'node_modules/popper.js/dist/umd/popper.min.js',
              'node_modules/bootstrap/dist/js/bootstrap.min.js',
              'node_modules/anchor-js/anchor.min.js'
            ],
            dest: 'pytorch_sphinx_theme/static/js/vendor',
            filter: 'isFile'
          }
        ]
      }
    },

    sass: {
      dev: {
        options: {
          style: 'expanded'
        },
        files: [{
          expand: true,
          cwd: 'scss',
          src: ['*.scss'],
          dest: 'pytorch_sphinx_theme/static/css',
          ext: '.css'
        }]
      },
      build: {
        options: {
          style: 'compressed'
        },
        files: [{
          expand: true,
          cwd: 'scss',
          src: ['*.scss'],
          dest: 'pytorch_sphinx_theme/static/css',
          ext: '.css'
        }]
      }
    },

    postcss: {
      options: {
        map: true,
        processors: [
          require("autoprefixer")({browsers: ["last 2 versions"]}),
        ]
      },

      dist: {
        files: {
          "pytorch_sphinx_theme/static/css/theme.css": "pytorch_sphinx_theme/static/css/theme.css"
        }
      }
    },

    browserify: {
      dev: {
        options: {
          external: ['jquery'],
          alias: {
            'pytorch-sphinx-theme': './js/theme.js'
          }
        },
        src: ['js/*.js'],
        dest: 'pytorch_sphinx_theme/static/js/theme.js'
      },
      build: {
        options: {
          external: ['jquery'],
          alias: {
            'pytorch-sphinx-theme': './js/theme.js'
          }
        },
        src: ['js/*.js'],
        dest: 'pytorch_sphinx_theme/static/js/theme.js'
      }
    },
    uglify: {
      dist: {
        options: {
          sourceMap: false,
          mangle: {
            reserved: ['jQuery'] // Leave 'jQuery' identifier unchanged
          },
          ie8: true // compliance with IE 6-8 quirks
        },
        files: [{
          expand: true,
          src: ['pytorch_sphinx_theme/static/js/*.js', '!pytorch_sphinx_theme/static/js/*.min.js'],
          dest: 'pytorch_sphinx_theme/static/js/',
          rename: function (dst, src) {
            // Use unminified file name for minified file
            return src;
          }
        }]
      }
    },
    exec: {
      build_sphinx: {
        cmd: 'sphinx-build ' + PROJECT_DIR + ' docs/build'
      }
    },
    clean: {
      build: ["docs/build"],
      fonts: ["pytorch_sphinx_theme/static/fonts"],
      images: ["pytorch_sphinx_theme/static/images"],
      css: ["pytorch_sphinx_theme/static/css"],
      js: ["pytorch_sphinx_theme/static/js/*", "!pytorch_sphinx_theme/static/js/modernizr.min.js"]
    },

    watch: {
      /* Compile scss changes into theme directory */
      sass: {
        files: ['scss/**/*.scss'],
        tasks: ['sass:dev', 'postcss:dist']
      },
      /* Changes in theme dir rebuild sphinx */
      sphinx: {
        files: ['pytorch_sphinx_theme/**/*', 'README.rst', 'docs/**/*.rst', 'docs/**/*.py'],
        tasks: ['clean:build','exec:build_sphinx']
      },
      /* JavaScript */
      browserify: {
        files: ['js/*.js'],
        tasks: ['browserify:dev']
      },
      /* live-reload the docs if sphinx re-builds */
      livereload: {
        files: ['docs/build/**/*'],
        options: { livereload: true }
      }
    }

  });

  grunt.loadNpmTasks('grunt-exec');
  grunt.loadNpmTasks('grunt-contrib-connect');
  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-contrib-sass');
  grunt.loadNpmTasks('grunt-contrib-clean');
  grunt.loadNpmTasks('grunt-contrib-copy');
  grunt.loadNpmTasks('grunt-open');
  grunt.loadNpmTasks('grunt-browserify');

  grunt.registerTask('default', ['clean','copy:fonts', 'copy:images', 'copy:vendor', 'sass:dev', 'postcss:dist', 'browserify:dev','exec:build_sphinx','connect','open','watch']);
  grunt.registerTask('build', ['clean','copy:fonts', 'copy:images', 'copy:vendor', 'sass:build', 'postcss:dist', 'browserify:build', 'uglify']);
}
