from collections import defaultdict
from bokeh.layouts import gridplot, column, row
from bokeh.models.widgets import Select
from bokeh.server.server import Server
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Panel, Tabs
from . import data as data_

def plot(source1d0, source1d1, source1d2):
    p1 = figure()
    p1.line(x='x', y='y_baseline', source=source1d0, color='blue', legend_label='baseline')
    p1.line(x='x', y='y_new', source=source1d0, color='red', legend_label='new')
    p1.xaxis.axis_label = 'Tensor size (power of 2)'
    p1.yaxis.axis_label = 'Performance (giga elements / second)'

    p2 = figure()
    p2.xaxis.axis_label = 'Tensor size (power of 2)'
    p2.yaxis.axis_label = 'Performance change (percentage)'
    p2.x_range = p1.x_range
    p2.vbar(x='positive_x', bottom=0, top='positive_y', color='green', width='width', source=source1d1)
    p2.vbar(x='negative_x', bottom=0, top='negative_y', color='red', width='width', source=source1d2)
    return gridplot([[p1, p2]])

def sort(l):
    try:
        return sorted(l, key=float)
    except ValueError:
        return sorted(l)

def plot_experiment(experiment):
    values = defaultdict(set)
    for setup, _ in experiment.items():
        for key, value in setup.items():
            values[key].add(value)

    selects = {}
    for k, v in values.items():
        v = sort(list(v))
        selects[k] = Select(title=k, value=v[0], options=v)

    def get_data():
        setup = {}
        for k, v in selects.items():
            setup[k] = v.value
        data = experiment[data_.hashabledict(setup)]

        baseline = data['baseline']
        new = data['new']
        compare = data['compare']
        x = [x.problem_size for x in baseline]
        y_baseline = [x.result for x in baseline]
        y_new = [x.result for x in new]
        y_compare = [x.result * 100 for x in compare]

        sx = sorted(x)
        if len(sx) >= 2:
            width = (sx[1] - sx[0]) * 0.8
        else:
            width = 2

        positive_x = []
        positive_y = []
        positive_width = []
        negative_x = []
        negative_y = []
        negative_width = []
        for x_, y_ in zip(x, y_compare):
            if y_ >= 0:
                positive_x.append(x_)
                positive_y.append(y_)
                positive_width.append(width)
            if y_ < 0:
                negative_x.append(x_)
                negative_y.append(y_)
                negative_width.append(width)

        return ({'x': x, 'y_baseline': y_baseline, 'y_new': y_new},
                {'positive_x': positive_x, 'positive_y': positive_y, 'width': positive_width},
                {'negative_x': negative_x, 'negative_y': negative_y, 'width': negative_width})

    d = get_data()
    source0 = ColumnDataSource(data=d[0])
    source1 = ColumnDataSource(data=d[1])
    source2 = ColumnDataSource(data=d[2])

    plot_ = plot(source0, source1, source2)

    def update(attrname, old, new):
        source0.data, source1.data, source2.data = get_data()

    for w in selects.values():
        w.on_change('value', update)

    plot_ = row(column(*selects.values()), plot_)

    return plot_

def serve(compare, port):
    def make_document(doc):
        tabs = []
        for title, experiment in compare.items():
            plot = plot_experiment(experiment)
            tab = Panel(child=plot, title=title)
            tabs.append(tab)
        tabs = Tabs(tabs=tabs)
        doc.title = 'Benchmark of TensorIterator'
        doc.add_root(tabs)

    print('listening port', port)
    server = Server(make_document, port=port)
    server.run_until_shutdown()
