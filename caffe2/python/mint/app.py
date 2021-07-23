## @package app
# Module caffe2.python.mint.app
import argparse
import flask
import glob
import numpy as np
import nvd3
import os
import sys
# pyre-fixme[21]: Could not find module `tornado.httpserver`.
import tornado.httpserver
# pyre-fixme[21]: Could not find a module corresponding to import `tornado.wsgi`
import tornado.wsgi

__folder__ = os.path.abspath(os.path.dirname(__file__))

app = flask.Flask(
    __name__,
    template_folder=os.path.join(__folder__, "templates"),
    static_folder=os.path.join(__folder__, "static")
)
args = None


def jsonify_nvd3(chart):
    chart.buildcontent()
    # Note(Yangqing): python-nvd3 does not seem to separate the built HTML part
    # and the script part. Luckily, it seems to be the case that the HTML part is
    # only a <div>, which can be accessed by chart.container; the script part,
    # while the script part occupies the rest of the html content, which we can
    # then find by chart.htmlcontent.find['<script>'].
    script_start = chart.htmlcontent.find('<script>') + 8
    script_end = chart.htmlcontent.find('</script>')
    return flask.jsonify(
        result=chart.container,
        script=chart.htmlcontent[script_start:script_end].strip()
    )


def visualize_summary(filename):
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        return 'Cannot load file {}: {}'.format(filename, str(e))
    chart_name = os.path.splitext(os.path.basename(filename))[0]
    chart = nvd3.lineChart(
        name=chart_name + '_summary_chart',
        height=args.chart_height,
        y_axis_format='.03g'
    )
    if args.sample < 0:
        step = max(data.shape[0] / -args.sample, 1)
    else:
        step = args.sample
    xdata = np.arange(0, data.shape[0], step)
    # data should have 4 dimensions.
    chart.add_serie(x=xdata, y=data[xdata, 0], name='min')
    chart.add_serie(x=xdata, y=data[xdata, 1], name='max')
    chart.add_serie(x=xdata, y=data[xdata, 2], name='mean')
    chart.add_serie(x=xdata, y=data[xdata, 2] + data[xdata, 3], name='m+std')
    chart.add_serie(x=xdata, y=data[xdata, 2] - data[xdata, 3], name='m-std')
    return jsonify_nvd3(chart)


def visualize_print_log(filename):
    try:
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[:, np.newaxis]
    except Exception as e:
        return 'Cannot load file {}: {}'.format(filename, str(e))
    chart_name = os.path.splitext(os.path.basename(filename))[0]
    chart = nvd3.lineChart(
        name=chart_name + '_log_chart',
        height=args.chart_height,
        y_axis_format='.03g'
    )
    if args.sample < 0:
        step = max(data.shape[0] / -args.sample, 1)
    else:
        step = args.sample
    xdata = np.arange(0, data.shape[0], step)
    # if there is only one curve, we also show the running min and max
    if data.shape[1] == 1:
        # We also print the running min and max for the steps.
        trunc_size = data.shape[0] / step
        running_mat = data[:trunc_size * step].reshape((trunc_size, step))
        chart.add_serie(
            x=xdata[:trunc_size],
            y=running_mat.min(axis=1),
            name='running_min'
        )
        chart.add_serie(
            x=xdata[:trunc_size],
            y=running_mat.max(axis=1),
            name='running_max'
        )
        chart.add_serie(x=xdata, y=data[xdata, 0], name=chart_name)
    else:
        for i in range(0, min(data.shape[1], args.max_curves)):
            # data should have 4 dimensions.
            chart.add_serie(
                x=xdata,
                y=data[xdata, i],
                name='{}[{}]'.format(chart_name, i)
            )

    return jsonify_nvd3(chart)


def visualize_file(filename):
    fullname = os.path.join(args.root, filename)
    if filename.endswith('summary'):
        return visualize_summary(fullname)
    elif filename.endswith('log'):
        return visualize_print_log(fullname)
    else:
        return flask.jsonify(
            result='Unsupport file: {}'.format(filename),
            script=''
        )


@app.route('/')
def index():
    files = glob.glob(os.path.join(args.root, "*.*"))
    files.sort()
    names = [os.path.basename(f) for f in files]
    return flask.render_template(
        'index.html',
        root=args.root,
        names=names,
        debug_messages=names
    )


@app.route('/visualization/<string:name>')
def visualization(name):
    ret = visualize_file(name)
    return ret


def main(argv):
    parser = argparse.ArgumentParser("The mint visualizer.")
    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=5000,
        help="The flask port to use."
    )
    parser.add_argument(
        '-r',
        '--root',
        type=str,
        default='.',
        help="The root folder to read files for visualization."
    )
    parser.add_argument(
        '--max_curves',
        type=int,
        default=5,
        help="The max number of curves to show in a dump tensor."
    )
    parser.add_argument(
        '--chart_height',
        type=int,
        default=300,
        help="The chart height for nvd3."
    )
    parser.add_argument(
        '-s',
        '--sample',
        type=int,
        default=-200,
        help="Sample every given number of data points. A negative "
        "number means the total points we will sample on the "
        "whole curve. Default 100 points."
    )
    global args
    args = parser.parse_args(argv)
    server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    server.listen(args.port)
    print("Tornado server starting on port {}.".format(args.port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main(sys.argv[1:])
